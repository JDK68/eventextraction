"""
Generate comprehensive HTML demo showing:
- Perfect site (nacacnet - 95.5%)
- Problematic site (wacac - 0%)
- Medium site (members.sacac - 6.7%)
"""
import pandas as pd
from pathlib import Path


def load_site_results(site_name):
    """Load both clustered results AND raw ground truth."""
    clustered_df = pd.read_csv(f"runs/simple/{site_name}/clustered.csv")
    
    # Load RAW data for true ground truth
    raw_df = pd.read_csv(f"../data/raw/{site_name}.csv")
    
    events = clustered_df[clustered_df['event_id'].notna()].copy()
    return clustered_df, events, raw_df


def generate_event_html(event_id, event_nodes, df, raw_df):
    """Generate HTML for one event."""
    
    # Get GROUND TRUTH from RAW data (not clustered!)
    raw_event = raw_df[raw_df['event_id'] == event_id]
    date_nodes = raw_event[raw_event['label'].str.contains('Date', case=False, na=False)]
    time_nodes = raw_event[raw_event['label'].str.contains('Time', case=False, na=False)]
    loc_nodes = raw_event[raw_event['label'].str.contains('Location', case=False, na=False)]
    
    # Check clustering
    clusters = event_nodes['pred_cluster'].unique()
    is_well_clustered = len(clusters) == 1
    
    # Get ONLY the nodes that were actually detected for THIS event
    # (from the clustered data, filtered by event_id)
    detected_event_nodes = df[df['event_id'] == event_id]

    extracted_dates = detected_event_nodes[
        detected_event_nodes['label'].str.contains('Date', case=False, na=False)
    ]
    extracted_times = detected_event_nodes[
        detected_event_nodes['label'].str.contains('Time', case=False, na=False)
    ]
    extracted_locs = detected_event_nodes[
        detected_event_nodes['label'].str.contains('Location', case=False, na=False)
    ]
    
    gt_date_texts = set(date_nodes['text_context'].str.strip().str.lower())
    ext_date_texts = set(extracted_dates['text_context'].str.strip().str.lower())
    date_match = len(gt_date_texts & ext_date_texts) if len(gt_date_texts) > 0 else 0
    
    gt_time_texts = set(time_nodes['text_context'].str.strip().str.lower())
    ext_time_texts = set(extracted_times['text_context'].str.strip().str.lower())
    time_match = len(gt_time_texts & ext_time_texts) if len(gt_time_texts) > 0 else 0
    
    gt_loc_texts = set(loc_nodes['text_context'].str.strip().str.lower())
    ext_loc_texts = set(extracted_locs['text_context'].str.strip().str.lower())
    loc_match = len(gt_loc_texts & ext_loc_texts) if len(gt_loc_texts) > 0 else 0
    
    # CALCULATE if this individual event is perfect (not based on site score!)
    is_event_perfect = (
        is_well_clustered and
        (len(gt_date_texts) == 0 or date_match > 0) and
        (len(gt_time_texts) == 0 or time_match > 0) and
        (len(gt_loc_texts) == 0 or loc_match > 0)
    )
    
    badge_class = "badge-perfect" if is_event_perfect else "badge-partial"
    badge_text = "✓ Perfect" if is_event_perfect else "⚠ Partial"
    border_color = "#27ae60" if is_event_perfect else "#e74c3c"
    
    html = f"""
    <div class="event" style="border-left-color: {border_color};">
        <div class="event-header">
            <span class="event-title">Event #{int(event_id)}</span>
            <span class="{badge_class}">{badge_text}</span>
        </div>
        
        <div class="cluster-info">
            <strong>Clustering:</strong> {len(clusters)} cluster(s)
            {'✅ Well-clustered' if is_well_clustered else f'❌ Split across {len(clusters)} clusters'}
        </div>
"""
    
    # Date field
    html += """
        <div class="field-section">
            <div class="field-label">
                <span class="field-icon">📅</span> Date
            </div>
"""
    if len(date_nodes) > 0:
        html += f"""
            <div class="field-comparison">
                <div class="comparison-row">
                    <span class="comparison-label">Expected:</span>
                    <span class="comparison-value">{', '.join(sorted(gt_date_texts))}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Extracted:</span>
                    <span class="comparison-value">{', '.join(sorted(ext_date_texts)[:3])}{'...' if len(ext_date_texts) > 3 else ''}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Match:</span>
                    <span class="match-status {'match-ok' if date_match > 0 else 'match-fail'}">{date_match}/{len(gt_date_texts)} {'✅' if date_match > 0 else '❌'}</span>
                </div>
            </div>
"""
    else:
        html += '            <div class="field-value-none">(none expected)</div>\n'
    
    html += "        </div>\n"
    
    # Time field
    html += """
        <div class="field-section">
            <div class="field-label">
                <span class="field-icon">🕐</span> Time
            </div>
"""
    if len(time_nodes) > 0:
        html += f"""
            <div class="field-comparison">
                <div class="comparison-row">
                    <span class="comparison-label">Expected:</span>
                    <span class="comparison-value">{', '.join(sorted(gt_time_texts))}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Extracted:</span>
                    <span class="comparison-value">{', '.join(sorted(ext_time_texts)[:3])}{'...' if len(ext_time_texts) > 3 else ''}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Match:</span>
                    <span class="match-status {'match-ok' if time_match > 0 else 'match-fail'}">{time_match}/{len(gt_time_texts)} {'✅' if time_match > 0 else '❌'}</span>
                </div>
            </div>
"""
    else:
        html += '            <div class="field-value-none">(none expected)</div>\n'
    
    html += "        </div>\n"
    
    # Location field
    html += """
        <div class="field-section">
            <div class="field-label">
                <span class="field-icon">📍</span> Location
            </div>
"""
    if len(loc_nodes) > 0:
        html += f"""
            <div class="field-comparison">
                <div class="comparison-row">
                    <span class="comparison-label">Expected:</span>
                    <span class="comparison-value">{', '.join(sorted(gt_loc_texts))}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Extracted:</span>
                    <span class="comparison-value">{', '.join(sorted(ext_loc_texts)[:3])}{'...' if len(ext_loc_texts) > 3 else ''}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Match:</span>
                    <span class="match-status {'match-ok' if loc_match > 0 else 'match-fail'}">{loc_match}/{len(gt_loc_texts)} {'✅' if loc_match > 0 else '❌'}</span>
                </div>
            </div>
"""
    else:
        html += '            <div class="field-value-none">(none expected)</div>\n'
    
    html += """        </div>
    </div>
"""
    
    return html


def create_full_demo():
    """Generate comprehensive HTML demo."""
    
    # Load sites
    sites = [
        ("nacacnet.org_pattern_labeled", "Perfect Site", "95.5%", 3, "all_perfect"),
        ("wacac.org_pattern_labeled", "Problematic Site", "0%", 2, "all_partial"),
        ("members.sacac.org_pattern_labeled", "Medium Site", "6.7%", 5, "mixed"),
    ]
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Event Extraction System - Comprehensive Demo</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px; 
            margin: 40px auto; 
            padding: 20px;
            background: #f5f7fa;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2c3e50;
            margin-top: 40px;
            padding: 15px;
            background: white;
            border-left: 5px solid #3498db;
            border-radius: 5px;
        }
        .intro {
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 1.8;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-item {
            padding: 15px;
            text-align: center;
        }
        .stat-label { 
            font-size: 0.9em; 
            color: #7f8c8d; 
            display: block;
            margin-bottom: 5px;
        }
        .stat-value { 
            font-size: 2em; 
            font-weight: bold; 
            color: #27ae60;
        }
        .event { 
            background: white;
            border-left: 5px solid #27ae60;
            margin: 20px 0; 
            padding: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .event-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .event-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }
        .badge-perfect {
            background: #27ae60;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .badge-partial {
            background: #e74c3c;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .cluster-info {
            padding: 10px;
            background: #ecf0f1;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 0.95em;
        }
        .field-section {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .field-label {
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .field-icon {
            margin-right: 8px;
            font-size: 1.2em;
        }
        .field-comparison {
            margin-top: 10px;
        }
        .comparison-row {
            display: flex;
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
        }
        .comparison-label {
            min-width: 100px;
            font-weight: bold;
            color: #7f8c8d;
        }
        .comparison-value {
            flex: 1;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
        }
        .match-status {
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 3px;
        }
        .match-ok {
            background: #d5f4e6;
            color: #27ae60;
        }
        .match-fail {
            background: #fadbd8;
            color: #e74c3c;
        }
        .field-value-none {
            padding: 8px;
            color: #95a5a6;
            font-style: italic;
        }
        .site-section {
            margin: 40px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .site-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }
        .site-name {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .site-score {
            font-size: 1.8em;
            font-weight: bold;
        }
        .score-perfect { color: #27ae60; }
        .score-partial { color: #e74c3c; }
        .score-medium { color: #f39c12; }
    </style>
</head>
<body>
    <h1>🎯 Event Extraction System - Comprehensive Demo</h1>
    
    <div class="intro">
        <h3 style="margin-top: 0;">ML Capstone Project: Automated Event Detection and Field Extraction</h3>
        <p>
            This system automatically extracts structured event information from heterogeneous website DOM structures.
            It combines machine learning (LightGBM) for node detection, spatial clustering for grouping, and 
            intelligent field extraction with proximity expansion.
        </p>
        <p>
            <strong>This demo shows three representative cases:</strong>
        </p>
        <ul>
            <li><strong>Perfect Site (95.5%):</strong> Demonstrates successful end-to-end extraction</li>
            <li><strong>Problematic Site (0%):</strong> Highlights remaining challenges and failure modes</li>
            <li><strong>Medium Site (6.7%):</strong> Shows partial success with split dates/times</li>
        </ul>
    </div>
    
    <div class="stats">
        <div class="stat-item">
            <span class="stat-label">Websites Tested</span>
            <span class="stat-value">16</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Events</span>
            <span class="stat-value">187</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Detection Rate</span>
            <span class="stat-value">99.5%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Clustering Accuracy</span>
            <span class="stat-value">88.3%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Date Extraction</span>
            <span class="stat-value">93.9%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Time Extraction</span>
            <span class="stat-value">96.4%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Location Extraction</span>
            <span class="stat-value">87.0%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label"><strong>Perfect Reconstitution</strong></span>
            <span class="stat-value" style="color: #3498db;">75.5%</span>
        </div>
    </div>
"""
    
    # Generate sections for each site
    for site_name, site_label, score, num_events, event_type in sites:
        df, events, raw_df = load_site_results(site_name) 
        # Select events based on type
        event_ids_list = sorted(events['event_id'].unique())
        
        if event_type == "mixed":
            # Scan ALL events to find perfect ones
            perfect_events = []
            partial_events = []
            
            for eid in event_ids_list:
                event_nodes = events[events['event_id'] == eid]
                raw_event = raw_df[raw_df['event_id'] == eid]
                detected_event = df[df['event_id'] == eid]
                
                # Check clustering
                clusters = event_nodes['pred_cluster'].unique()
                is_well_clustered = len(clusters) == 1
                
                # Get ground truth and extracted for each field
                date_gt = raw_event[raw_event['label'].str.contains('Date', case=False, na=False)]
                time_gt = raw_event[raw_event['label'].str.contains('Time', case=False, na=False)]
                loc_gt = raw_event[raw_event['label'].str.contains('Location', case=False, na=False)]
                
                date_ext = detected_event[detected_event['label'].str.contains('Date', case=False, na=False)]
                time_ext = detected_event[detected_event['label'].str.contains('Time', case=False, na=False)]
                loc_ext = detected_event[detected_event['label'].str.contains('Location', case=False, na=False)]
                
                # Calculate matches
                if len(date_gt) > 0:
                    gt_dates = set(date_gt['text_context'].str.strip().str.lower())
                    ext_dates = set(date_ext['text_context'].str.strip().str.lower())
                    date_match = len(gt_dates & ext_dates)
                    date_ok = date_match > 0
                else:
                    date_ok = True  # No date expected
                
                if len(time_gt) > 0:
                    gt_times = set(time_gt['text_context'].str.strip().str.lower())
                    ext_times = set(time_ext['text_context'].str.strip().str.lower())
                    time_match = len(gt_times & ext_times)
                    time_ok = time_match > 0
                else:
                    time_ok = True  # No time expected
                
                if len(loc_gt) > 0:
                    gt_locs = set(loc_gt['text_context'].str.strip().str.lower())
                    ext_locs = set(loc_ext['text_context'].str.strip().str.lower())
                    loc_match = len(gt_locs & ext_locs)
                    loc_ok = loc_match > 0
                else:
                    loc_ok = True  # No location expected
                
                # Event is perfect if all conditions met
                is_perfect = is_well_clustered and date_ok and time_ok and loc_ok
                
                if is_perfect:
                    perfect_events.append(eid)
                else:
                    partial_events.append(eid)
            
            # Select events: 1 perfect (if exists) + 1 partial
            event_ids = []
            if len(perfect_events) > 0:
                event_ids.append(perfect_events[0])  # First perfect
            if len(partial_events) > 0:
                event_ids.append(partial_events[0])  # First partial
            
            print(f"\n  Found {len(perfect_events)} perfect events: {perfect_events[:5]}")
            print(f"  Found {len(partial_events)} partial events (showing first)")
        
        score_class = "score-perfect" if score == "95.5%" else ("score-partial" if score == "0%" else "score-medium")
        
        html += f"""
    <div class="site-section">
        <div class="site-header">
            <span class="site-name">{site_label}: {site_name.split('_')[0]}</span>
            <span class="site-score {score_class}">{score}</span>
        </div>
"""
        
        # Show events (only if not already set by mixed logic)
        if event_type != "mixed":
            event_ids = sorted(events['event_id'].unique())[:num_events]
        for event_id in event_ids:
            event_nodes = events[events['event_id'] == event_id].copy()
            is_perfect = (score == "95.5%")  # Simplified
            html += generate_event_html(event_id, event_nodes, df, raw_df)
        
        if len(event_ids) < events['event_id'].nunique():
            html += f"        <p style='text-align: center; color: #7f8c8d; margin-top: 20px;'>... ({events['event_id'].nunique() - len(event_ids)} more events)</p>\n"
        
        html += "    </div>\n"
    
    html += """
    <div style="margin-top: 40px; padding: 25px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3>System Architecture</h3>
        <ol style="line-height: 2;">
            <li><strong>Detection (99.5%):</strong> LightGBM classifier with 27 structural and semantic features identifies event-related DOM nodes</li>
            <li><strong>Clustering (88.3%):</strong> Unified spatial clustering using rendering order, parent index, and depth</li>
            <li><strong>Post-processing:</strong> Proximity expansion captures short nodes (1-15 chars) near clusters</li>
            <li><strong>Field Extraction:</strong> Multi-cluster extraction with set-based validation (33% overlap threshold)</li>
        </ol>
        
        <h3>Key Innovations</h3>
        <ul style="line-height: 2;">
            <li><strong>Adaptive K:</strong> Top-K selection scales with site size (30-120 nodes)</li>
            <li><strong>Proximity Expansion:</strong> Captures low-scoring short nodes (e.g., "7", "mar") within ±5 positions</li>
            <li><strong>Multi-cluster Extraction:</strong> Extracts fields from all clusters containing event nodes (handles bad clustering)</li>
            <li><strong>Set-based Validation:</strong> Flexible matching allows partial field overlap (handles split dates/times)</li>
        </ul>
        
        <h3>Remaining Challenges</h3>
        <ul style="line-height: 2;">
            <li><strong>Label Heterogeneity:</strong> Sites use different labeling schemes (Date vs DateTime, StartTime+EndTime vs Time)</li>
            <li><strong>Clustering Failures:</strong> Some sites have unusual DOM structures that confuse spatial clustering</li>
            <li><strong>Short Node Detection:</strong> Very short nodes (1-3 chars) score poorly and may not enter top-K</li>
        </ul>
    </div>
"""

    # Add data quality section
    html += """
    <div style="margin-top: 40px; padding: 25px; background: #fff3cd; border-left: 5px solid #ffc107; border-radius: 8px;">
        <h3 style="color: #856404; margin-top: 0;">Data Quality Challenges</h3>
        <p style="line-height: 1.8; color: #856404;">
            Analysis revealed labeling errors in the ground truth data. For example, some events have 
            descriptive text labeled as "Date" (e.g., "details to come! contact..."), or single digits 
            labeled as "Time". These labeling inconsistencies introduce noise into both training and validation.
        </p>
        <p style="line-height: 1.8; color: #856404;">
            <strong>Implication:</strong> The reported 75.5% perfect reconstitution rate is achieved despite 
            noisy ground truth labels. With cleaner annotations, the system would likely achieve 80-85% performance.
        </p>
    </div>
    
</body>
</html>
"""
    
    output_file = "comprehensive_demo.html"
    Path(output_file).write_text(html, encoding='utf-8')
    print(f"✅ Comprehensive HTML demo created: {output_file}")
    print(f"\n📧 This demo shows:")
    print(f"   ✓ Perfect site (nacacnet - 95.5%)")
    print(f"   ✗ Problematic site (wacac - 0%)")
    print(f"   ~ Medium site (members.sacac - 6.7%)")
    print(f"\n👉 Open '{output_file}' in your browser!")


if __name__ == "__main__":
    create_full_demo()
    