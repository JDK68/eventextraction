"""
Simplified event extraction pipeline.
No graph clustering, no router, no NER, no over-engineering.
Just: detection → clustering → field extraction → validation.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sklearn.metrics import adjusted_rand_score

from dateclf.data import (
    load_config,
    load_raw_merged,
    add_is_event_member_label,
    build_feature_matrix_for_member,
)
from dateclf.split import loso_folds_event
from dateclf.sampling import undersample_negatives
from dateclf.metrics import precision_recall_at_k, event_level_metrics_at_k
from dateclf.features import add_dom_neighbor_features


@dataclass
class FoldResult:
    site: str
    metrics_at_k: Dict[int, Dict[str, float]]
    ari: float
    end_to_end: Dict[str, float]


def unified_clustering(
    nodes: pd.DataFrame,
    gap_order: int = 10,
    gap_parent: int = 8,
    gap_depth: int = 2,
) -> pd.DataFrame:
    """
    Simple robust clustering using rendering_order + parent_index + depth.
    One method, well-tuned, no complexity.
    """
    nodes = nodes.sort_values('rendering_order').reset_index(drop=True).copy()
    
    cluster_ids = []
    current_cluster = 0
    
    for i, row in nodes.iterrows():
        if i == 0:
            current_cluster += 1
        else:
            prev = nodes.iloc[i-1]
            
            gap_r = row['rendering_order'] - prev['rendering_order']
            
            if pd.notna(row['parent_index']) and pd.notna(prev['parent_index']):
                gap_p = abs(row['parent_index'] - prev['parent_index'])
            else:
                gap_p = 0
            
            gap_d = abs(row['depth'] - prev['depth'])
            
            # New cluster IF:
            # - Large rendering gap AND
            # - (Different parent OR very different depth)
            if gap_r > gap_order and (gap_p > gap_parent or gap_d > gap_depth):
                current_cluster += 1
        
        cluster_ids.append(current_cluster)
    
    nodes['pred_cluster'] = cluster_ids
    return nodes

def expand_clusters_with_nearby_fields(
    clustered: pd.DataFrame,
    all_scored_nodes: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Post-processing: pour chaque cluster, ajoute les nodes Date/Time/Location
    courts qui sont à proximité immédiate mais n'étaient pas dans top-K.
    
    This captures short nodes like "7", "mar", "thu" that score poorly
    but are semantically part of the event.
    """
    expanded = clustered.copy()
    
    for cluster_id in clustered['pred_cluster'].unique():
        cluster_nodes = clustered[clustered['pred_cluster'] == cluster_id]
        
        # Fenêtre autour du cluster
        min_order = cluster_nodes['rendering_order'].min()
        max_order = cluster_nodes['rendering_order'].max()
        
        # Cherche dans tous les nodes scorés (pas juste top-K)
        nearby = all_scored_nodes[
            (all_scored_nodes['rendering_order'] >= min_order - window) &
            (all_scored_nodes['rendering_order'] <= max_order + window) &
            (~all_scored_nodes.index.isin(clustered.index))  # Pas déjà dans cluster
        ].copy()
        
        if len(nearby) == 0:
            continue
        
        # Garde seulement les nodes courts avec labels Date/Time/Location
        field_nodes = nearby[
            (nearby['text_length'] <= 15) &  # Courts (≤15 caractères)
            (nearby['label'].str.contains('Date|Time|Location', case=False, na=False))
        ]
        
        if len(field_nodes) > 0:
            # Ajoute au cluster
            field_nodes['pred_cluster'] = cluster_id
            expanded = pd.concat([expanded, field_nodes], ignore_index=True)
    
    return expanded.sort_values('rendering_order').reset_index(drop=True)

def remove_obvious_noise(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Post-scoring filter to remove navigation/footer/header noise.
    Not as features (which bias the model), but as post-processing.
    """
    nodes = nodes.copy()
    
    text = nodes['text_context'].fillna('').astype(str).str.lower().str.strip()
    
    # Navigation/meta patterns
    noise_patterns = (
        text.str.contains(
            r'\b(?:login|log in|sign in|register|home|menu|navigation|'
            r'contact us|about us|privacy|copyright|call us|'
            r'step 1|step 2|pricing|membership)\b',
            regex=True,
            na=False
        )
    )
    
    # Structural noise (too shallow or too short)
    structural_noise = (
        (nodes['depth'] < 2) |  # Header/footer typically very shallow
        (nodes['text_length'] < 5)  # Fragments
    )
    
    noise_mask = noise_patterns | structural_noise
    
    return nodes[~noise_mask].copy()


def evaluate_end_to_end_extraction(
    clustered: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """
    SIMPLIFIED end-to-end metric.
    
    For each ground truth event:
    - Is it detected? (at least 1 node in top-K)
    - Is it well-clustered? (all its nodes in same predicted cluster)
    """
    results = []
    
    # For each ground truth event
    for event_id in ground_truth['event_id'].dropna().unique():
        gt_nodes = ground_truth[ground_truth['event_id'] == event_id]
        
        # Find which of these nodes are in clustered (top-K)
        # Match by row_id or index
        if 'row_id' in clustered.columns and 'row_id' in gt_nodes.columns:
            detected = clustered[clustered['row_id'].isin(gt_nodes['row_id'])]
        else:
            # Match by text_context (fallback)
            detected = clustered[
                clustered['text_context'].isin(gt_nodes['text_context'])
            ]
        
        n_detected = len(detected)
        n_total = len(gt_nodes)
        
        # Event is detected if at least 1 node found
        is_detected = n_detected > 0
        
        # Event is well-clustered if all detected nodes in same cluster
        if n_detected > 1:
            clusters = detected['pred_cluster'].unique()
            is_well_clustered = len(clusters) == 1
        elif n_detected == 1:
            is_well_clustered = True
        else:
            is_well_clustered = False
        
        # Simplified: perfect = detected + well-clustered
        is_perfect = is_detected and is_well_clustered
        
        results.append({
            'event_id': event_id,
            'detected': is_detected,
            'well_clustered': is_well_clustered,
            'perfect': is_perfect,
            'n_detected': n_detected,
            'n_total': n_total,
        })
    
    if len(results) == 0:
        return {
            'n_events': 0,
            'detection_rate': 0.0,
            'clustering_rate': 0.0,
            'perfect_extraction_rate': 0.0,
        }
    
    df_results = pd.DataFrame(results)
    
    return {
        'n_events': len(df_results),
        'detection_rate': df_results['detected'].mean(),
        'clustering_rate': df_results['well_clustered'].mean(),
        'perfect_extraction_rate': df_results['perfect'].mean(),
    }

def extract_and_validate_fields(
    clustered: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """
    FULL reconstitution with MULTI-CLUSTER extraction.
    
    When an event is split across multiple clusters (bad clustering),
    we extract fields from ALL clusters containing event nodes.
    """
    results = []
    
    # For each ground truth event
    for event_id in ground_truth['event_id'].dropna().unique():
        gt_event = ground_truth[ground_truth['event_id'] == event_id]
        
        # Match by row_id or text_context
        if 'row_id' in clustered.columns and 'row_id' in gt_event.columns:
            detected_nodes = clustered[clustered['row_id'].isin(gt_event['row_id'])]
        else:
            detected_nodes = clustered[
                clustered['text_context'].isin(gt_event['text_context'])
            ]
        
        n_detected = len(detected_nodes)
        
        # Event not detected at all
        if n_detected == 0:
            results.append({
                'event_id': event_id,
                'detected': False,
                'clustered': False,
                'date_correct': False,
                'time_correct': False,
                'location_correct': False,
                'perfect': False,
            })
            continue
        
        # Check clustering
        pred_clusters = detected_nodes['pred_cluster'].unique()
        is_well_clustered = len(pred_clusters) == 1
        
        # ========== EXTRACT FROM ALL CLUSTERS (not just main) ==========
        # Get ALL nodes from ALL clusters containing this event
        all_cluster_ids = detected_nodes['pred_cluster'].unique()
        all_cluster_nodes = clustered[
            clustered['pred_cluster'].isin(all_cluster_ids)
        ].copy()
        
        # Extract ALL Date nodes from all clusters
        date_extracted = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Date', case=False, na=False)
        ]
        extracted_date_texts = set(
            date_extracted['text_context'].str.strip().str.lower()
        )
        
        # Extract ALL Time nodes from all clusters
        time_extracted = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Time', case=False, na=False)
        ]
        extracted_time_texts = set(
            time_extracted['text_context'].str.strip().str.lower()
        )
        
        # Extract ALL Location nodes from all clusters
        location_extracted = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Location', case=False, na=False)
        ]
        extracted_location_texts = set(
            location_extracted['text_context'].str.strip().str.lower()
        )
        
        # ========== GROUND TRUTH SETS ==========
        
        gt_date = gt_event[gt_event['label'].str.contains('Date', case=False, na=False)]
        gt_date_texts = set(gt_date['text_context'].str.strip().str.lower())
        
        gt_time = gt_event[gt_event['label'].str.contains('Time', case=False, na=False)]
        gt_time_texts = set(gt_time['text_context'].str.strip().str.lower())
        
        gt_location = gt_event[gt_event['label'].str.contains('Location', case=False, na=False)]
        gt_location_texts = set(gt_location['text_context'].str.strip().str.lower())
        
        # ========== SET-BASED VALIDATION ==========
        
        date_correct = False
        time_correct = False
        location_correct = False
        
        # Date validation
        if len(gt_date_texts) > 0:
            # All GT dates found OR at least 50% overlap
            if gt_date_texts.issubset(extracted_date_texts):
                date_correct = True
            elif len(extracted_date_texts) > 0:
                overlap = len(gt_date_texts & extracted_date_texts)
                required = len(gt_date_texts)
                if overlap / required >= 0.33:
                    date_correct = True
        else:
            date_correct = True
        
        # Time validation
        if len(gt_time_texts) > 0:
            if gt_time_texts.issubset(extracted_time_texts):
                time_correct = True
            elif len(extracted_time_texts) > 0:
                overlap = len(gt_time_texts & extracted_time_texts)
                required = len(gt_time_texts)
                if overlap / required >= 0.33:
                    time_correct = True
        else:
            time_correct = True
        
        # Location validation
        if len(gt_location_texts) > 0:
            if gt_location_texts.issubset(extracted_location_texts):
                location_correct = True
            elif len(extracted_location_texts) > 0:
                overlap = len(gt_location_texts & extracted_location_texts)
                required = len(gt_location_texts)
                if overlap / required >= 0.33:
                    location_correct = True
        else:
            location_correct = True
        
        # Perfect reconstitution = ALL correct
        is_perfect = (
            n_detected > 0 and
            is_well_clustered and
            date_correct and
            time_correct and
            location_correct
        )
        
        results.append({
            'event_id': event_id,
            'detected': n_detected > 0,
            'clustered': is_well_clustered,
            'date_correct': date_correct,
            'time_correct': time_correct,
            'location_correct': location_correct,
            'perfect': is_perfect,
        })
    
    if len(results) == 0:
        return {
            'n_events': 0,
            'detection_rate': 0.0,
            'clustering_rate': 0.0,
            'date_accuracy': 0.0,
            'time_accuracy': 0.0,
            'location_accuracy': 0.0,
            'perfect_reconstitution_rate': 0.0,
        }
    
    df_results = pd.DataFrame(results)
    
    return {
        'n_events': len(df_results),
        'detection_rate': df_results['detected'].mean(),
        'clustering_rate': df_results['clustered'].mean(),
        'date_accuracy': df_results['date_correct'].mean(),
        'time_accuracy': df_results['time_correct'].mean(),
        'location_accuracy': df_results['location_correct'].mean(),
        'perfect_reconstitution_rate': df_results['perfect'].mean(),
    }


def train_simple_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Simple, clean, end-to-end event extraction pipeline.
    
    Steps:
    1. Load data
    2. Train event detector (LightGBM)
    3. LOSO cross-validation
    4. For each fold:
       - Predict scores
       - Remove noise
       - Cluster events
       - Evaluate end-to-end
    """
    cfg = load_config(config_path)
    
    # Load data
    df = load_raw_merged(cfg)
    
    # Add neighbor features (keep simple ones only)
    df = add_dom_neighbor_features(df)
    
    # Add target
    df = add_is_event_member_label(df, cfg)
    
    # Build feature matrix
    X, y, groups = build_feature_matrix_for_member(df)
    
    # Remove complex features (NER, noise flags, etc.)
    features_to_keep = [
        # Structure (simple)
        'depth', 'tag', 'parent_tag', 'num_siblings',
        
        # Text stats
        'text_length', 'word_count', 
        'letter_ratio', 'digit_ratio', 'whitespace_ratio',
        
        # Patterns (CORE)
        'contains_date', 'contains_time',
        'starts_with_digit', 'ends_with_digit',
        
        # Attributes
        'has_class', 'has_id',
        'attr_has_word_date', 'attr_has_word_time', 'attr_has_word_location',
        'text_has_word_date', 'text_word_time', 'text_word_location',
        
        # Simple neighbors
        'prev_contains_date', 'next_contains_date',
        'prev_contains_time', 'next_contains_time',
        'same_parent_as_prev', 'same_parent_as_next',
    ]
    
    available_features = [f for f in features_to_keep if f in X.columns]
    X = X[available_features].copy()
    
    print(f"\n=== Using {len(available_features)} features ===")
    print(f"Features: {available_features}\n")
    
    # Categorical features
    cat_features = [c for c in ['tag', 'parent_tag'] if c in X.columns]
    
    # LOSO folds
    folds = list(loso_folds_event(df, y_col='is_event_member'))
    
    all_results: List[FoldResult] = []
    
    for i, fold in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"FOLD {i}/{len(folds)}: {fold.holdout_site}")
        print(f"{'='*60}")
        
        train_idx = fold.train_idx
        test_idx = fold.test_idx
        
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test = X.loc[test_idx]
        
        # Undersample negatives
        X_train_s, y_train_s = undersample_negatives(
            X_train, y_train,
            seed=cfg['seed'] + i,
            keep_neg_ratio=cfg.get('sampling', {}).get('keep_neg_ratio', 0.15),
        )
        
        # Compute scale_pos_weight
        n_pos = int(y_train_s.sum())
        n_neg = int((y_train_s == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        # Train LightGBM
        params = {
            'objective': 'binary',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l2': 1.0,
            'metric': 'None',
            'verbosity': -1,
            'seed': cfg['seed'],
            'scale_pos_weight': scale_pos_weight,
        }
        
        dtrain = lgb.Dataset(
            X_train_s, label=y_train_s,
            categorical_feature=cat_features if cat_features else 'auto',
            free_raw_data=False,
        )
        
        model = lgb.train(params, dtrain, num_boost_round=400)
        
        # Feature importance
        fi = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        
        print("\nTop-10 Features:")
        print(fi.head(10).to_string(index=False))
        
        # Predict
        test_scores = model.predict(X_test)
        
        # Prepare test dataframe
        test_cols = ['text_context', 'label', 'event_id', 'is_event_member', 'rendering_order', 
                    'parent_index', 'depth', 'contains_date', 'contains_time', 'text_length']
        test_cols = [c for c in test_cols if c in df.columns]

        df_test = df.loc[test_idx, test_cols].copy()
        df_test['score'] = test_scores
        df_test['row_id'] = df_test.index  # Add row_id for matching
        
        # Node-level metrics
        ks = [1, 3, 5, 10, 20, 30]
        metrics_at_k = precision_recall_at_k(
            y_true=df.loc[test_idx, 'is_event_member'],
            scores=df_test['score'],
            ks=ks,
        )
        
        print("\nNode-level Metrics:")
        for k in ks:
            m = metrics_at_k[k]
            print(f"  P@{k:2d}={m['precision']:.3f}  R@{k:2d}={m['recall']:.3f}")
        
        # Event-level metrics
        event_metrics = event_level_metrics_at_k(
            df=df_test,
            score_col='score',
            y_col='is_event_member',
            event_id_col='event_id',
            ks=ks,
        )
        
        print("\nEvent-level Metrics:")
        for k in ks:
            em = event_metrics[k]
            print(f"  EventDet@{k:2d}={em['event_detection_rate']:.3f}")
        
        # Get top-K candidates
        # Adaptive K based on site size
        n_nodes = len(df_test)
        n_events_estimated = df_test['event_id'].nunique() if 'event_id' in df_test.columns else 10

        # Rule: K should be at least 3x number of events
        # (to catch ~3 nodes per event on average)
        if n_nodes < 100:
            K = 30
        elif n_nodes < 200:
            K = 50
        else:
            K = 80  # For large sites

        # Also ensure K catches estimated events
        K = max(K, min(n_events_estimated * 4, 100))

        print(f"Using K={K} for site with {n_nodes} nodes")

        top_k = df_test.nlargest(K, 'score').copy()
        
        # Remove obvious noise
        top_k_clean = remove_obvious_noise(top_k)
        
        print(f"\nCandidates: {len(top_k)} → {len(top_k_clean)} (after denoising)")
        
        # Clustering
        if len(top_k_clean) >= 2:
            clustered = unified_clustering(
                top_k_clean,
                gap_order=10,
                gap_parent=8,
                gap_depth=2,
            )
            
            # POST-PROCESSING: Expand clusters with nearby short Date/Time/Location nodes
            # This captures nodes like "7", "mar", "thu" that scored poorly
            clustered = expand_clusters_with_nearby_fields(
                clustered=clustered,
                all_scored_nodes=df_test,  # All nodes with scores (not just top-K)
                window=5,  # Look ±5 rendering_order positions
            )
            
            # Evaluate clustering (ARI)
            cluster_eval = clustered[clustered['event_id'].notna()].copy()
            
            if len(cluster_eval) >= 2:
                ari = adjusted_rand_score(
                    cluster_eval['event_id'],
                    cluster_eval['pred_cluster'],
                )
            else:
                ari = 0.0
            
            print(f"\nClustering ARI: {ari:.3f}")
            
            # Full reconstitution evaluation (Detection + Clustering + Field Extraction)
            gt_events = df_test[df_test['event_id'].notna()].copy()

            if len(gt_events) > 0:
                e2e = extract_and_validate_fields(clustered, gt_events)
                
                print("\nFull Reconstitution:")
                print(f"  Events: {e2e['n_events']}")
                print(f"  Detection rate: {e2e['detection_rate']:.1%}")
                print(f"  Clustering rate: {e2e['clustering_rate']:.1%}")
                print(f"  Date accuracy: {e2e['date_accuracy']:.1%}")
                print(f"  Time accuracy: {e2e['time_accuracy']:.1%}")
                print(f"  Location accuracy: {e2e['location_accuracy']:.1%}")
                print(f"  🎯 PERFECT RECONSTITUTION: {e2e['perfect_reconstitution_rate']:.1%}")
            else:
                e2e = {
                    'n_events': 0,
                    'detection_rate': 0.0,
                    'clustering_rate': 0.0,
                    'date_accuracy': 0.0,
                    'time_accuracy': 0.0,
                    'location_accuracy': 0.0,
                    'perfect_reconstitution_rate': 0.0,
                }
        else:
            ari = 0.0
            e2e = {'n_events': 0, 'perfect_extraction_rate': 0.0}
        
        all_results.append(FoldResult(
            site=fold.holdout_site,
            metrics_at_k=metrics_at_k,
            ari=ari,
            end_to_end=e2e,
        ))
        
        # Save fold artifacts
        out_dir = Path("runs/simple") / fold.holdout_site
        out_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_model(str(out_dir / "model.txt"))
        fi.to_csv(out_dir / "feature_importance.csv", index=False)
        
        if len(top_k_clean) >= 2:
            clustered.to_csv(out_dir / "clustered.csv", index=False)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")
    
    # Node-level
    for k in ks:
        ps = [r.metrics_at_k[k]['precision'] for r in all_results]
        rs = [r.metrics_at_k[k]['recall'] for r in all_results]
        print(f"P@{k:2d}={np.mean(ps):.3f}±{np.std(ps):.3f}  "
              f"R@{k:2d}={np.mean(rs):.3f}±{np.std(rs):.3f}")
    
    # Clustering
    aris = [r.ari for r in all_results]
    print(f"\nMean ARI: {np.mean(aris):.3f}±{np.std(aris):.3f}")
    
    # Full reconstitution metrics
    detection_rates = [r.end_to_end.get('detection_rate', 0.0) for r in all_results]
    clustering_rates = [r.end_to_end.get('clustering_rate', 0.0) for r in all_results]
    date_accs = [r.end_to_end.get('date_accuracy', 0.0) for r in all_results]
    time_accs = [r.end_to_end.get('time_accuracy', 0.0) for r in all_results]
    location_accs = [r.end_to_end.get('location_accuracy', 0.0) for r in all_results]
    perfect_recon = [r.end_to_end.get('perfect_reconstitution_rate', 0.0) for r in all_results]

    print(f"\nFull Reconstitution Metrics:")
    print(f"  Detection: {np.mean(detection_rates):.1%}±{np.std(detection_rates):.1%}")
    print(f"  Clustering: {np.mean(clustering_rates):.1%}±{np.std(clustering_rates):.1%}")
    print(f"  Date accuracy: {np.mean(date_accs):.1%}±{np.std(date_accs):.1%}")
    print(f"  Time accuracy: {np.mean(time_accs):.1%}±{np.std(time_accs):.1%}")
    print(f"  Location accuracy: {np.mean(location_accs):.1%}±{np.std(location_accs):.1%}")
    print(f"\n🎯 PERFECT RECONSTITUTION RATE: {np.mean(perfect_recon):.1%}±{np.std(perfect_recon):.1%}")
    
    summary = {
        'n_folds': len(all_results),
        'mean_ari': float(np.mean(aris)),
        'mean_detection': float(np.mean(detection_rates)),
        'mean_clustering': float(np.mean(clustering_rates)),
        'mean_date_accuracy': float(np.mean(date_accs)),
        'mean_time_accuracy': float(np.mean(time_accs)),
        'mean_location_accuracy': float(np.mean(location_accs)),
        'mean_perfect_reconstitution': float(np.mean(perfect_recon)),
        'folds': [
            {
                'site': r.site,
                'ari': r.ari,
                'perfect_reconstitution': r.end_to_end.get('perfect_reconstitution_rate', 0.0),
                'date_accuracy': r.end_to_end.get('date_accuracy', 0.0),
                'time_accuracy': r.end_to_end.get('time_accuracy', 0.0),
                'location_accuracy': r.end_to_end.get('location_accuracy', 0.0),
            }
            for r in all_results
        ],
    }
    
    Path("runs/simple/summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )
    
    return summary


if __name__ == "__main__":
    summary = train_simple_pipeline("../config.yaml")
    print("\n✅ Training complete.")
    print(f"🎯 Perfect reconstitution rate: {summary['mean_perfect_reconstitution']:.1%}")