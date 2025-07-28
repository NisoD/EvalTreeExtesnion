#!/usr/bin/env python3
"""
Debug Sunburst Data Structure

Identifies why the HTML files are showing empty sunbursts
"""

import json
import statistics
from collections import defaultdict

def load_data():
    """Load DOVE scores and MMLU tree structure"""
    print("Loading data files...")
    
    # Load DOVE scores
    with open('MMLU_DOVE.json', 'r') as f:
        dove_scores = json.load(f)
    
    # Load MMLU.json structure (handle truncation)
    try:
        with open('MMLU.json', 'r') as f:
            content = f.read()
            eval_tree = json.loads(content)
    except json.JSONDecodeError:
        with open('MMLU.json', 'r') as f:
            content = f.read()
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
            eval_tree = json.loads(content)
    
    print(f"Loaded {len(dove_scores)} DOVE scores")
    print(f"Loaded EvalTree with size {eval_tree.get('size', 'unknown')}")
    
    return dove_scores, eval_tree

def collect_question_indices(node):
    """Recursively collect all question indices from a subtree"""
    indices = []
    
    if isinstance(node, dict):
        if 'subtrees' in node:
            subtrees = node['subtrees']
            
            if isinstance(subtrees, int):
                indices.append(subtrees)
            elif isinstance(subtrees, list):
                for subtree in subtrees:
                    indices.extend(collect_question_indices(subtree))
            elif isinstance(subtrees, dict):
                indices.extend(collect_question_indices(subtrees))
    
    return indices

def extract_simple_hierarchical_data(node, dove_scores, path="", level=0, max_levels=3):
    """Extract hierarchical data with simpler structure for debugging"""
    data = []
    
    if not isinstance(node, dict) or level >= max_levels:
        return data
    
    capability = node.get('capability', f'Level_{level}')
    
    # Much shorter capability names
    if len(capability) > 20:
        capability_short = capability[:17] + "..."
    else:
        capability_short = capability
    
    # Get question indices for this node
    question_indices = collect_question_indices(node)
    
    # Calculate DOVE statistics
    subtree_scores = []
    for idx in question_indices:
        if str(idx) in dove_scores:
            subtree_scores.append(dove_scores[str(idx)])
    
    if subtree_scores and len(subtree_scores) >= 3:  # Only include if we have enough data
        mean_score = statistics.mean(subtree_scores)
        question_count = len(subtree_scores)
        
        # Create SIMPLE path structure
        if level == 0:
            # Root level
            node_id = capability_short
            parent_id = ""
        else:
            # Child levels - use simple naming
            node_id = f"{path}_{capability_short}" if path else capability_short
            parent_id = path
        
        # Add data point
        data.append({
            'ids': node_id,
            'labels': capability_short,
            'parents': parent_id,
            'values': question_count,
            'dove_score': round(mean_score, 3),
            'level': level,
            'question_count': question_count,
            'full_capability': capability
        })
        
        print(f"Level {level}: {capability_short} -> {question_count} questions, DOVE: {mean_score:.3f}")
    
    # Recurse into subtrees with simpler path
    if 'subtrees' in node and isinstance(node['subtrees'], list) and level < max_levels - 1:
        current_path = f"{path}_{capability_short}" if path else capability_short
        for subtree in node['subtrees']:
            data.extend(extract_simple_hierarchical_data(subtree, dove_scores, current_path, level + 1, max_levels))
    
    return data

def create_simple_sunburst_test():
    """Create a very simple test sunburst to verify functionality"""
    import plotly.graph_objects as go
    
    # Simple test data
    test_data = [
        {'ids': 'Root', 'labels': 'Root', 'parents': '', 'values': 100},
        {'ids': 'A', 'labels': 'Category A', 'parents': 'Root', 'values': 60},
        {'ids': 'B', 'labels': 'Category B', 'parents': 'Root', 'values': 40},
        {'ids': 'A1', 'labels': 'Sub A1', 'parents': 'A', 'values': 30},
        {'ids': 'A2', 'labels': 'Sub A2', 'parents': 'A', 'values': 30},
        {'ids': 'B1', 'labels': 'Sub B1', 'parents': 'B', 'values': 40}
    ]
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in test_data],
        labels=[d['labels'] for d in test_data],
        parents=[d['parents'] for d in test_data],
        values=[d['values'] for d in test_data],
        branchvalues="total"
    ))
    
    fig.update_layout(
        title="Simple Test Sunburst",
        font_size=12,
        width=600,
        height=600
    )
    
    fig.write_html("test_sunburst.html")
    print("✅ Test sunburst saved to test_sunburst.html")
    return fig

def debug_data_structure(dove_scores, eval_tree):
    """Debug the data structure to find issues"""
    print("\n" + "="*80)
    print("DEBUGGING DATA STRUCTURE")
    print("="*80)
    
    # Extract hierarchical data
    print("Extracting hierarchical data...")
    hierarchical_data = extract_simple_hierarchical_data(eval_tree, dove_scores)
    
    print(f"\nExtracted {len(hierarchical_data)} data points")
    
    if not hierarchical_data:
        print("❌ No hierarchical data extracted!")
        return None
    
    # Check data structure
    print("\nFirst 10 data points:")
    for i, item in enumerate(hierarchical_data[:10]):
        print(f"{i+1:2d}. ID: '{item['ids']}' | Parent: '{item['parents']}' | Label: '{item['labels']}' | Values: {item['values']}")
    
    # Check for issues
    print("\nChecking for common issues...")
    
    # Check for empty parents
    root_nodes = [d for d in hierarchical_data if d['parents'] == '']
    print(f"Root nodes: {len(root_nodes)}")
    
    # Check for circular references
    ids = set(d['ids'] for d in hierarchical_data)
    parents = set(d['parents'] for d in hierarchical_data if d['parents'])
    orphaned_parents = parents - ids
    if orphaned_parents:
        print(f"⚠️ Orphaned parents (parents without corresponding IDs): {orphaned_parents}")
    
    # Check for duplicate IDs
    id_counts = defaultdict(int)
    for d in hierarchical_data:
        id_counts[d['ids']] += 1
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    if duplicates:
        print(f"⚠️ Duplicate IDs: {duplicates}")
    
    return hierarchical_data

def create_fixed_sunburst(hierarchical_data):
    """Create sunburst with fixed data structure"""
    import plotly.graph_objects as go
    
    if not hierarchical_data:
        print("❌ No data to create sunburst")
        return None
    
    # Sort by weakness (lowest DOVE score first) and take top 50
    sorted_data = sorted(hierarchical_data, key=lambda x: x['dove_score'])[:50]
    
    print(f"Creating sunburst with {len(sorted_data)} weakest categories")
    
    # Create color mapping
    min_score = min(d['dove_score'] for d in sorted_data)
    max_score = max(d['dove_score'] for d in sorted_data)
    
    colors = []
    for d in sorted_data:
        if max_score > min_score:
            norm_score = (d['dove_score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 0
        colors.append(norm_score)
    
    print(f"Score range: {min_score:.3f} - {max_score:.3f}")
    print(f"First few items:")
    for i, item in enumerate(sorted_data[:5]):
        print(f"  {i+1}. {item['labels']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in sorted_data],
        labels=[f"{d['labels']}<br>{d['dove_score']}" for d in sorted_data],
        parents=[d['parents'] for d in sorted_data],
        values=[d['values'] for d in sorted_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[0]}<br>' +
                     'Full Name: %{customdata[1]}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['full_capability'][:50] + "..." if len(d['full_capability']) > 50 else d['full_capability']] for d in sorted_data],
        marker=dict(
            colorscale='RdYlBu_r',
            cmin=0,
            cmax=1,
            colors=colors,
            line=dict(color="white", width=2)
        ),
        maxdepth=3
    ))
    
    fig.update_layout(
        title=f"DOVE Weakness Profile - Top {len(sorted_data)} Weakest<br><sub>Red = Most Weak, Blue = Less Weak</sub>",
        font_size=12,
        width=800,
        height=800
    )
    
    fig.write_html("fixed_dove_sunburst.html")
    print("✅ Fixed sunburst saved to fixed_dove_sunburst.html")
    
    return fig

def main():
    """Main debugging function"""
    print("Starting Sunburst Debug Analysis...")
    
    try:
        # Create simple test first
        print("Creating simple test sunburst...")
        create_simple_sunburst_test()
        
        # Load and debug real data
        dove_scores, eval_tree = load_data()
        hierarchical_data = debug_data_structure(dove_scores, eval_tree)
        
        if hierarchical_data:
            # Create fixed sunburst
            print("\nCreating fixed sunburst...")
            create_fixed_sunburst(hierarchical_data)
        
        print("\n" + "="*80)
        print("DEBUG ANALYSIS COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • test_sunburst.html - Simple test to verify plotly works")
        print("  • fixed_dove_sunburst.html - Fixed DOVE weakness sunburst")
        print("\nOpen these files in your browser to check if they display correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 