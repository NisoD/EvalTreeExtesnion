#!/usr/bin/env python3
"""
Shortened Capability Sunburst

Uses the shortened tree.json directly without any capability shortening
Red = Low accuracy (weak), Blue = High accuracy (strong)
"""

import json
import plotly.graph_objects as go
import statistics
from collections import defaultdict

def load_shortened_tree():
    """Load the shortened tree JSON"""
    print("Loading shortened tree...")
    
    try:
        with open('shortened tree.json', 'r') as f:
            content = f.read()
            tree = json.loads(content)
    except json.JSONDecodeError:
        # Handle potential truncation
        with open('shortened tree.json', 'r') as f:
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
            tree = json.loads(content)
    
    print(f"Loaded shortened tree with size {tree.get('size', 'unknown')}")
    return tree

def extract_shortened_hierarchical_data(node, path="", level=0, max_levels=4):
    """Extract hierarchical data from shortened tree structure"""
    data = []
    
    if not isinstance(node, dict) or level >= max_levels:
        return data
    
    # Get capability (already shortened)
    capability = node.get('capability', f'Level_{level}')
    
    # Get weakness stats if available
    weakness_stats = node.get('weakness_stats', {})
    
    if weakness_stats and weakness_stats.get('question_count', 0) >= 3:
        mean_score = weakness_stats.get('mean_dove_score', 0.5)
        question_count = weakness_stats.get('question_count', 0)
        
        # Determine weakness level
        if mean_score < 0.3:
            weakness_level = "Critical"
        elif mean_score < 0.5:
            weakness_level = "High"
        elif mean_score < 0.7:
            weakness_level = "Moderate"
        else:
            weakness_level = "Low"
        
        # Create path for sunburst
        current_path = [p for p in path.split(" > ") if p] + [capability]
        
        # Add data point
        data.append({
            'ids': " > ".join(current_path),
            'labels': capability,  # Use shortened capability directly
            'parents': " > ".join(current_path[:-1]) if len(current_path) > 1 else "",
            'values': question_count,
            'dove_score': round(mean_score, 3),
            'weakness_level': weakness_level,
            'color_value': mean_score,
            'level': level,
            'question_count': question_count,
            'std_score': weakness_stats.get('std_dove_score', 0),
            'coverage': weakness_stats.get('coverage', 0)
        })
    
    # Recurse into subtrees
    if 'subtrees' in node and isinstance(node['subtrees'], list):
        for subtree in node['subtrees']:
            new_path = f"{path} > {capability}" if path else capability
            data.extend(extract_shortened_hierarchical_data(subtree, new_path, level + 1, max_levels))
    
    return data

def create_shortened_sunburst(hierarchical_data, title="DOVE Weakness Profile - Shortened Capabilities", top_n=30):
    """Create sunburst with shortened capabilities and correct color mapping"""
    
    # Filter and sort by weakness (lowest DOVE score first)
    filtered_data = [d for d in hierarchical_data if d['question_count'] >= 3]
    sorted_data = sorted(filtered_data, key=lambda x: x['dove_score'])
    
    # Take the weakest top_n categories
    top_weak_data = sorted_data[:top_n]
    
    if not top_weak_data:
        print("❌ No data after filtering!")
        return None, None
    
    print(f"Creating sunburst with {len(top_weak_data)} categories")
    print("Top 5 weakest categories:")
    for i, item in enumerate(top_weak_data[:5]):
        print(f"  {i+1}. {item['labels']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Color mapping - RED = LOW ACCURACY (WEAK), BLUE = HIGH ACCURACY (STRONG)
    min_score = min(d['dove_score'] for d in top_weak_data)
    max_score = max(d['dove_score'] for d in top_weak_data)
    
    colors = []
    for d in top_weak_data:
        if max_score > min_score:
            # Normalize: 0 = red (lowest accuracy/weakest), 1 = blue (highest accuracy/strongest)
            norm_score = (d['dove_score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 0  # All same, make red (weak)
        colors.append(norm_score)
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in top_weak_data],
        labels=[f"{d['labels']}<br>{d['dove_score']}" for d in top_weak_data],
        parents=[d['parents'] for d in top_weak_data],
        values=[d['values'] for d in top_weak_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[0]}<br>' +
                     'Weakness Level: %{customdata[1]}<br>' +
                     'Std Dev: %{customdata[2]:.3f}<br>' +
                     'Coverage: %{customdata[3]:.1%}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['weakness_level'], 
                    d['std_score'], d['coverage']] for d in top_weak_data],
        marker=dict(
            colorscale='RdYlBu_r',  # Red to Blue (reversed)
            cmin=0,
            cmax=1,
            colors=colors,
            colorbar=dict(
                title=dict(
                    text="Accuracy<br>Level",
                    font=dict(size=12, family="Arial", color="black")
                ),
                tickmode="array",
                tickvals=[0, 0.33, 0.66, 1.0],
                ticktext=["Low", "Moderate", "High", "Strong"],
                tickfont=dict(size=10, family="Arial", color="black"),
                len=0.6,
                thickness=12,
                x=1.02
            ),
            line=dict(color="white", width=1.5)
        ),
        maxdepth=4,
        textfont=dict(size=9, family="Arial", color="black"),
        insidetextorientation='horizontal'
    ))
    
    # Academic paper styling
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:11px; color:#666'>Analysis of {len(top_weak_data)} Capability Clusters (Pre-shortened)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=15, family="Arial", color="black"),
            pad=dict(t=20, b=20)
        ),
        font=dict(size=10, family="Arial", color="black"),
        width=750,
        height=750,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=120)
    )
    
    return fig, top_weak_data

def print_shortened_results_summary(weakness_data):
    """Print summary for shortened capabilities"""
    print("\n" + "="*60)
    print("SHORTENED CAPABILITIES SUNBURST - RESULTS SUMMARY")
    print("="*60)
    
    # Group by weakness level
    by_level = defaultdict(list)
    for item in weakness_data:
        by_level[item['weakness_level']].append(item)
    
    total_questions = sum(item['values'] for item in weakness_data)
    print(f"\nAnalyzed: {len(weakness_data)} capability clusters")
    print(f"Questions: {total_questions} total")
    
    for level in ['Critical', 'High', 'Moderate', 'Low']:
        if level in by_level:
            items = by_level[level]
            level_questions = sum(item['values'] for item in items)
            avg_score = sum(item['dove_score'] for item in items) / len(items)
            
            print(f"\n{level}: {len(items)} clusters ({level_questions} questions, avg={avg_score:.3f})")
            
            # Show top examples with shortened names
            for i, item in enumerate(items[:3], 1):
                print(f"  {i}. {item['labels']} ({item['dove_score']})")

def main():
    """Main execution"""
    print("🚀 Creating Shortened Capabilities Sunburst...")
    print("="*60)
    
    try:
        # Load shortened tree
        tree = load_shortened_tree()
        
        # Extract hierarchical data (no shortening needed)
        print("Extracting hierarchical data from shortened tree...")
        hierarchical_data = extract_shortened_hierarchical_data(tree)
        
        if not hierarchical_data:
            print("❌ No hierarchical data extracted!")
            return
        
        print(f"Extracted {len(hierarchical_data)} hierarchical data points")
        
        # Create main sunburst
        print("Creating main sunburst...")
        fig, weakness_data = create_shortened_sunburst(hierarchical_data, top_n=25)
        
        if fig:
            fig.write_html("shortened_capabilities_sunburst.html")
            print("✅ Main sunburst saved to shortened_capabilities_sunburst.html")
        
        # Create extended version
        print("Creating extended version...")
        fig_ext, weakness_data_ext = create_shortened_sunburst(hierarchical_data, 
                                                              title="DOVE Weakness Analysis - Extended (Shortened)",
                                                              top_n=40)
        if fig_ext:
            fig_ext.write_html("shortened_capabilities_extended_sunburst.html")
            print("✅ Extended version saved to shortened_capabilities_extended_sunburst.html")
        
        # Print results
        if weakness_data:
            print_shortened_results_summary(weakness_data)
        
        print("\n" + "="*60)
        print("🎉 SHORTENED CAPABILITIES SUNBURST COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  • shortened_capabilities_sunburst.html - Main figure (25 categories)")
        print("  • shortened_capabilities_extended_sunburst.html - Extended (40 categories)")
        print("\n📊 Features:")
        print("  • Uses pre-shortened capabilities directly")
        print("  • Red = Low accuracy (weak), Blue = High accuracy (strong)")
        print("  • Academic styling for paper")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 