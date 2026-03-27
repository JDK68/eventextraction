"""
Visualize event extraction results.
Shows how the system reconstructs events.
"""
import pandas as pd
from pathlib import Path
import sys


def visualize_event_extraction(site_name: str, num_events: int = 3):
    """Show event extraction results."""
    
    # Load clustered results
    results_path = Path(f"runs/simple/{site_name}/clustered.csv")
    
    if not results_path.exists():
        print(f"❌ No results found for {site_name}")
        print(f"   Looking for: {results_path}")
        print("\n📁 Available sites:")
        for d in Path("runs/simple").iterdir():
            if d.is_dir() and (d / "clustered.csv").exists():
                print(f"   ✓ {d.name}")
        return
    
    df = pd.read_csv(results_path)
    
    # Get events with ground truth
    events = df[df['event_id'].notna()].copy()
    
    if len(events) == 0:
        print(f"⚠️  No events found in {site_name}")
        return
    
    print("\n" + "="*80)
    print(f"🎯 EVENT EXTRACTION RESULTS: {site_name}")
    print("="*80)
    print(f"\nTotal events: {events['event_id'].nunique()}")
    print(f"Total event nodes: {len(events)}")
    
    # Group by event
    event_ids = sorted(events['event_id'].unique())[:num_events]
    
    for event_id in event_ids:
        event_nodes = events[events['event_id'] == event_id].copy()
        
        print("\n" + "-"*80)
        print(f"📌 EVENT #{int(event_id)}")
        print("-"*80)
        
        # Clusters
        clusters = event_nodes['pred_cluster'].unique()
        print(f"\n🔗 Clustering: {len(clusters)} cluster(s)")
        if len(clusters) == 1:
            print(f"   ✅ Well-clustered (all nodes in cluster #{int(clusters[0])})")
        else:
            print(f"   ⚠️  Split across clusters: {[int(c) for c in sorted(clusters)]}")
        
        # Show nodes by label
        print(f"\n📝 Nodes ({len(event_nodes)} total):")
        
        for label in sorted(event_nodes['label'].unique()):
            label_nodes = event_nodes[event_nodes['label'] == label]
            print(f"\n   {label}:")
            for _, node in label_nodes.iterrows():
                cluster = int(node['pred_cluster'])
                text = node['text_context']
                score = node.get('score', 0.0)
                
                # Truncate long text
                if len(text) > 60:
                    text = text[:57] + "..."
                
                print(f"      [Cluster #{cluster}] Score:{score:.3f} → \"{text}\"")
        
        # Extract fields (multi-cluster)
        print(f"\n🎯 FIELD EXTRACTION:")
        
        all_cluster_ids = event_nodes['pred_cluster'].unique()
        all_cluster_nodes = df[df['pred_cluster'].isin(all_cluster_ids)].copy()
        
        # Date
        gt_date = event_nodes[event_nodes['label'].str.contains('Date', case=False, na=False)]
        extracted_date = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Date', case=False, na=False)
        ]
        
        print("\n   📅 DATE:")
        if len(gt_date) > 0:
            gt_texts = sorted(set(gt_date['text_context'].str.strip().str.lower()))
            ext_texts = sorted(set(extracted_date['text_context'].str.strip().str.lower()))
            
            print(f"      Expected:  {gt_texts}")
            print(f"      Extracted: {ext_texts}")
            
            overlap = len(set(gt_texts) & set(ext_texts))
            if overlap > 0:
                print(f"      ✅ Match: {overlap}/{len(gt_texts)}")
            else:
                print(f"      ❌ No match")
        else:
            print(f"      (none expected)")
        
        # Time
        gt_time = event_nodes[event_nodes['label'].str.contains('Time', case=False, na=False)]
        extracted_time = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Time', case=False, na=False)
        ]
        
        print("\n   🕐 TIME:")
        if len(gt_time) > 0:
            gt_texts = sorted(set(gt_time['text_context'].str.strip().str.lower()))
            ext_texts = sorted(set(extracted_time['text_context'].str.strip().str.lower()))
            
            print(f"      Expected:  {gt_texts}")
            print(f"      Extracted: {ext_texts}")
            
            overlap = len(set(gt_texts) & set(ext_texts))
            if overlap > 0:
                print(f"      ✅ Match: {overlap}/{len(gt_texts)}")
            else:
                print(f"      ❌ No match")
        else:
            print(f"      (none expected)")
        
        # Location
        gt_loc = event_nodes[event_nodes['label'].str.contains('Location', case=False, na=False)]
        extracted_loc = all_cluster_nodes[
            all_cluster_nodes['label'].str.contains('Location', case=False, na=False)
        ]
        
        print("\n   📍 LOCATION:")
        if len(gt_loc) > 0:
            gt_texts = sorted(set(gt_loc['text_context'].str.strip().str.lower()))
            ext_texts = sorted(set(extracted_loc['text_context'].str.strip().str.lower()))
            
            print(f"      Expected:  {gt_texts}")
            print(f"      Extracted: {ext_texts}")
            
            overlap = len(set(gt_texts) & set(ext_texts))
            if overlap > 0:
                print(f"      ✅ Match: {overlap}/{len(gt_texts)}")
            else:
                print(f"      ❌ No match")
        else:
            print(f"      (none expected)")
        
        # Verdict
        print("\n" + "─"*80)
        is_perfect = (
            len(clusters) == 1 and
            (len(gt_date) == 0 or len(set(gt_date['text_context'].str.lower()) & set(extracted_date['text_context'].str.lower())) > 0) and
            (len(gt_time) == 0 or len(set(gt_time['text_context'].str.lower()) & set(extracted_time['text_context'].str.lower())) > 0) and
            (len(gt_loc) == 0 or len(set(gt_loc['text_context'].str.lower()) & set(extracted_loc['text_context'].str.lower())) > 0)
        )
        
        if is_perfect:
            print("   🎉 PERFECT RECONSTITUTION ✅")
        else:
            print("   ⚠️  PARTIAL RECONSTITUTION")
    
    if len(event_ids) < events['event_id'].nunique():
        print(f"\n... ({events['event_id'].nunique() - len(event_ids)} more events)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        site = sys.argv[1]
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        visualize_event_extraction(site, num)
    else:
        print("Usage: python view_results.py SITE_NAME [NUM_EVENTS]")
        print("\nAvailable sites:")
        for d in sorted(Path("runs/simple").iterdir()):
            if d.is_dir() and (d / "clustered.csv").exists():
                print(f"  {d.name}")