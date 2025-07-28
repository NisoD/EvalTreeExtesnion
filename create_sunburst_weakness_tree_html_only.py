#!/usr/bin/env python3
"""
DOVE-Based Sunburst Weakness Tree (HTML Only)

Creates interactive sunburst visualizations showing hierarchical clustering
with DOVE-based weakness profiling compared to original MMLU structure.
HTML output only to avoid image export dependencies.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statistics
from collections import defaultdict
import numpy as np

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

def extract_hierarchical_data(node, dove_scores, path="", level=0, max_levels=4):
    """Extract hierarchical data for sunburst visualization"""
    data = []
    
    if not isinstance(node, dict) or level >= max_levels:
        return data
    
    capability = node.get('capability', f'Level_{level}')
    
    # Shorten capability names for readability
    if len(capability) > 35:
        capability_short = capability[:32] + "..."
    else:
        capability_short = capability
    
    # Get question indices for this node
    question_indices = collect_question_indices(node)
    
    # Calculate DOVE statistics
    subtree_scores = []
    for idx in question_indices:
        if str(idx) in dove_scores:
            subtree_scores.append(dove_scores[str(idx)])
    
    if subtree_scores:
        mean_score = statistics.mean(subtree_scores)
        question_count = len(subtree_scores)
        
        # Determine weakness level and color
        if mean_score < 0.3:
            weakness_level = "Critical"
            color_value = 1
        elif mean_score < 0.5:
            weakness_level = "High"
            color_value = 2
        elif mean_score < 0.7:
            weakness_level = "Moderate"
            color_value = 3
        else:
            weakness_level = "Low"
            color_value = 4
        
        # Create path for sunburst
        current_path = [p for p in path.split(" > ") if p] + [capability_short]
        
        # Add data point
        data.append({
            'ids': " > ".join(current_path),
            'labels': capability_short,
            'parents': " > ".join(current_path[:-1]) if len(current_path) > 1 else "",
            'values': question_count,
            'dove_score': round(mean_score, 3),
            'weakness_level': weakness_level,
            'color_value': color_value,
            'level': level,
            'question_count': question_count,
            'full_capability': capability
        })
    
    # Recurse into subtrees
    if 'subtrees' in node and isinstance(node['subtrees'], list):
        for subtree in node['subtrees']:
            new_path = f"{path} > {capability_short}" if path else capability_short
            data.extend(extract_hierarchical_data(subtree, dove_scores, new_path, level + 1, max_levels))
    
    return data

def create_dove_sunburst(hierarchical_data, title="DOVE-Based Weakness Profile"):
    """Create DOVE-based sunburst visualization"""
    
    # Filter for meaningful nodes (at least 3 questions)
    filtered_data = [d for d in hierarchical_data if d['question_count'] >= 3]
    
    if not filtered_data:
        print("No data available for sunburst visualization")
        return None
    
    # Create color scale based on weakness levels
    color_discrete_map = {
        1: '#d32f2f',  # Critical - Red
        2: '#f57c00',  # High - Orange  
        3: '#fbc02d',  # Moderate - Yellow
        4: '#388e3c'   # Low - Green
    }
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in filtered_data],
        labels=[f"{d['labels']}<br>{d['dove_score']}" for d in filtered_data],
        parents=[d['parents'] for d in filtered_data],
        values=[d['values'] for d in filtered_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[0]}<br>' +
                     'Weakness: %{customdata[1]}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['weakness_level']] for d in filtered_data],
        marker=dict(
            colors=[color_discrete_map[d['color_value']] for d in filtered_data],
            line=dict(color="white", width=2)
        ),
        maxdepth=4
    ))
    
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Color: Red=Critical, Orange=High, Yellow=Moderate, Green=Low Weakness</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        font_size=11,
        width=900,
        height=900
    )
    
    return fig

def extract_original_hierarchy(node, path="", level=0, max_levels=4):
    """Extract original MMLU hierarchy for comparison"""
    data = []
    
    if not isinstance(node, dict) or level >= max_levels:
        return data
    
    capability = node.get('capability', f'Level_{level}')
    
    # Shorten capability names
    if len(capability) > 35:
        capability_short = capability[:32] + "..."
    else:
        capability_short = capability
    
    size = node.get('size', 0)
    
    if size > 0:
        # Create path for sunburst
        current_path = [p for p in path.split(" > ") if p] + [capability_short]
        
        # Add data point
        data.append({
            'ids': " > ".join(current_path),
            'labels': capability_short,
            'parents': " > ".join(current_path[:-1]) if len(current_path) > 1 else "",
            'values': size,
            'level': level,
            'full_capability': capability
        })
    
    # Recurse into subtrees
    if 'subtrees' in node and isinstance(node['subtrees'], list):
        for subtree in node['subtrees']:
            new_path = f"{path} > {capability_short}" if path else capability_short
            data.extend(extract_original_hierarchy(subtree, new_path, level + 1, max_levels))
    
    return data

def create_original_sunburst(hierarchical_data, title="Original MMLU Structure"):
    """Create original MMLU sunburst visualization"""
    
    # Filter for meaningful nodes
    filtered_data = [d for d in hierarchical_data if d['values'] >= 10]
    
    if not filtered_data:
        print("No data available for original sunburst")
        return None
    
    # Create sunburst with consistent colors by level
    colors = px.colors.qualitative.Set3
    level_colors = {}
    for d in filtered_data:
        if d['level'] not in level_colors:
            level_colors[d['level']] = colors[d['level'] % len(colors)]
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in filtered_data],
        labels=[d['labels'] for d in filtered_data],
        parents=[d['parents'] for d in filtered_data],
        values=[d['values'] for d in filtered_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Size: %{value}<br>' +
                     '<extra></extra>',
        marker=dict(
            colors=[level_colors[d['level']] for d in filtered_data],
            line=dict(color="white", width=2)
        ),
        maxdepth=4
    ))
    
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Hierarchical Structure by Size</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        font_size=11,
        width=900,
        height=900
    )
    
    return fig

def create_comparative_sunburst(dove_data, original_data):
    """Create side-by-side comparison of DOVE vs Original"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("DOVE-Based Weakness Profile", "Original MMLU Structure")
    )
    
    # Filter data
    dove_filtered = [d for d in dove_data if d['question_count'] >= 3]
    original_filtered = [d for d in original_data if d['values'] >= 10]
    
    # DOVE sunburst colors
    color_discrete_map = {
        1: '#d32f2f',  # Critical
        2: '#f57c00',  # High
        3: '#fbc02d',  # Moderate
        4: '#388e3c'   # Low
    }
    
    # Add DOVE sunburst
    if dove_filtered:
        fig.add_trace(go.Sunburst(
            ids=[d['ids'] for d in dove_filtered],
            labels=[f"{d['labels']}<br>{d['dove_score']}" for d in dove_filtered],
            parents=[d['parents'] for d in dove_filtered],
            values=[d['values'] for d in dove_filtered],
            branchvalues="total",
            marker=dict(
                colors=[color_discrete_map[d['color_value']] for d in dove_filtered],
                line=dict(color="white", width=1)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Questions: %{value}<br>' +
                         'DOVE Score: %{customdata[0]}<br>' +
                         'Weakness: %{customdata[1]}<br>' +
                         '<extra></extra>',
            customdata=[[d['dove_score'], d['weakness_level']] for d in dove_filtered],
            maxdepth=3,
            domain=dict(column=0)
        ), row=1, col=1)
    
    # Original sunburst colors
    colors = px.colors.qualitative.Set3
    level_colors = {}
    for d in original_filtered:
        if d['level'] not in level_colors:
            level_colors[d['level']] = colors[d['level'] % len(colors)]
    
    # Add original sunburst
    if original_filtered:
        fig.add_trace(go.Sunburst(
            ids=[d['ids'] for d in original_filtered],
            labels=[d['labels'] for d in original_filtered],
            parents=[d['parents'] for d in original_filtered],
            values=[d['values'] for d in original_filtered],
            branchvalues="total",
            marker=dict(
                colors=[level_colors[d['level']] for d in original_filtered],
                line=dict(color="white", width=1)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Size: %{value}<br>' +
                         '<extra></extra>',
            maxdepth=3,
            domain=dict(column=1)
        ), row=1, col=2)
    
    fig.update_layout(
        title={
            'text': "DOVE Weakness Profile vs Original MMLU Structure",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        font_size=10,
        width=1800,
        height=900,
        annotations=[
            dict(text="Red=Critical, Orange=High, Yellow=Moderate, Green=Low", 
                 x=0.25, y=0.02, xref="paper", yref="paper", showarrow=False, font_size=11),
            dict(text="Colors by hierarchical level", 
                 x=0.75, y=0.02, xref="paper", yref="paper", showarrow=False, font_size=11)
        ]
    )
    
    return fig

def create_weakness_summary_table(dove_data):
    """Create summary table of weaknesses"""
    
    # Filter and sort by weakness
    filtered_data = [d for d in dove_data if d['question_count'] >= 3]
    sorted_data = sorted(filtered_data, key=lambda x: x['dove_score'])
    
    # Create summary by weakness level
    weakness_summary = defaultdict(list)
    for d in sorted_data:
        weakness_summary[d['weakness_level']].append(d)
    
    print("\n" + "="*100)
    print("SUNBURST WEAKNESS SUMMARY")
    print("="*100)
    
    for level in ['Critical', 'High', 'Moderate', 'Low']:
        if level in weakness_summary:
            items = weakness_summary[level]
            print(f"\n{level.upper()} WEAKNESSES ({len(items)} clusters):")
            print("-" * 80)
            
            for i, item in enumerate(items[:10], 1):  # Show top 10
                capability = item['full_capability'][:60] + "..." if len(item['full_capability']) > 60 else item['full_capability']
                print(f"{i:2d}. {capability}")
                print(f"    DOVE Score: {item['dove_score']} | Questions: {item['question_count']}")
                if i < len(items):
                    print()
            
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more {level.lower()} weakness clusters")
    
    return weakness_summary

def main():
    """Main execution function"""
    print("Creating DOVE-Based Sunburst Weakness Tree (HTML Only)...")
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract hierarchical data
        print("Extracting DOVE-based hierarchical data...")
        dove_data = extract_hierarchical_data(eval_tree, dove_scores)
        
        print("Extracting original hierarchical data...")
        original_data = extract_original_hierarchy(eval_tree)
        
        if not dove_data:
            print("No DOVE data could be extracted")
            return
        
        print(f"Extracted {len(dove_data)} DOVE capability clusters")
        print(f"Extracted {len(original_data)} original capability clusters")
        
        # Create individual sunburst visualizations
        print("Creating DOVE sunburst visualization...")
        dove_fig = create_dove_sunburst(dove_data)
        if dove_fig:
            dove_fig.write_html("dove_sunburst_weakness.html")
            print("DOVE sunburst saved to dove_sunburst_weakness.html")
        
        print("Creating original structure sunburst...")
        original_fig = create_original_sunburst(original_data)
        if original_fig:
            original_fig.write_html("original_mmlu_sunburst.html")
            print("Original sunburst saved to original_mmlu_sunburst.html")
        
        # Create comparative visualization
        print("Creating comparative sunburst visualization...")
        comparative_fig = create_comparative_sunburst(dove_data, original_data)
        if comparative_fig:
            comparative_fig.write_html("comparative_sunburst.html")
            print("Comparative sunburst saved to comparative_sunburst.html")
        
        # Create summary table
        weakness_summary = create_weakness_summary_table(dove_data)
        
        print("\n" + "="*100)
        print("SUNBURST VISUALIZATION COMPLETE!")
        print("="*100)
        print("Generated files:")
        print("  • dove_sunburst_weakness.html - Interactive DOVE weakness sunburst")
        print("  • original_mmlu_sunburst.html - Original MMLU structure sunburst")
        print("  • comparative_sunburst.html - Side-by-side comparison")
        print("\nOpen the .html files in your browser for interactive visualization!")
        print("You can click on segments to drill down into the hierarchy!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 