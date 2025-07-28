#!/usr/bin/env python3
"""
Working Shortened Sunburst - Based on Minimal Working Approach
Uses unique IDs and simple structure like the minimal version that worked
"""

import json
import plotly.graph_objects as go
import statistics
import hashlib
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

def create_unique_id(text, level, index):
    """Create a unique ID that won't have duplicates"""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"L{level}_{index}_{text_hash}"

def extract_simple_shortened_data(tree):
    """Extract data using simple approach like minimal working version"""
    print("Extracting data using simple approach...")
    data = []
    node_index = 0
    
    def traverse(node, level=0, parent_id=""):
        nonlocal node_index
        
        if not isinstance(node, dict) or level > 3:  # Limit depth
            return
        
        capability = node.get('capability', f'Node_{node_index}')
        weakness_stats = node.get('weakness_stats', {})
        
        if weakness_stats and weakness_stats.get('question_count', 0) >= 3:
            mean_score = weakness_stats.get('mean_dove_score', 0.5)
            question_count = weakness_stats.get('question_count', 0)
            
            # Create unique ID
            unique_id = create_unique_id(capability, level, node_index)
            
            # Determine weakness level
            if mean_score < 0.3:
                weakness_level = "Critical"
            elif mean_score < 0.5:
                weakness_level = "High"
            elif mean_score < 0.7:
                weakness_level = "Moderate"
            else:
                weakness_level = "Low"
            
            # Add data point with simple structure
            data.append({
                'ids': unique_id,
                'labels': f"{capability}<br>{mean_score:.3f}",
                'parents': parent_id,
                'values': question_count,
                'dove_score': round(mean_score, 3),
                'weakness_level': weakness_level,
                'level': level,
                'question_count': question_count,
                'std_score': weakness_stats.get('std_dove_score', 0),
                'coverage': weakness_stats.get('coverage', 0),
                'full_capability': capability
            })
            
            print(f"Added: {capability[:30]}... (Score: {mean_score:.3f}, Questions: {question_count})")
            
            # Recurse into subtrees with this node as parent
            if 'subtrees' in node and isinstance(node['subtrees'], list):
                for subtree in node['subtrees']:
                    node_index += 1
                    traverse(subtree, level + 1, unique_id)
        
        node_index += 1
    
    # Start traversal
    traverse(tree)
    return data

def create_working_shortened_sunburst(data, title="DOVE Weakness Profile - Shortened", top_n=25):
    """Create sunburst using working approach"""
    
    if not data:
        print("❌ No data available for sunburst")
        return None, None
    
    # Sort by weakness and take top N
    sorted_data = sorted(data, key=lambda x: x['dove_score'])[:top_n]
    
    print(f"Creating working sunburst with {len(sorted_data)} categories")
    print("Top 5 weakest categories:")
    for i, item in enumerate(sorted_data[:5]):
        print(f"  {i+1}. {item['full_capability'][:30]}... (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Verify no duplicate IDs
    ids = [d['ids'] for d in sorted_data]
    if len(ids) != len(set(ids)):
        print("⚠️ Warning: Duplicate IDs detected, fixing...")
        for i, item in enumerate(sorted_data):
            item['ids'] = f"node_{i}_{item['ids']}"
    
    # Color mapping - RED = LOW ACCURACY (WEAK), BLUE = HIGH ACCURACY (STRONG)
    min_score = min(d['dove_score'] for d in sorted_data)
    max_score = max(d['dove_score'] for d in sorted_data)
    
    colors = []
    for d in sorted_data:
        if max_score > min_score:
            # Normalize: 0 = red (lowest accuracy/weakest), 1 = blue (highest accuracy/strongest)
            norm_score = (d['dove_score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 0  # All same, make red (weak)
        colors.append(norm_score)
    
    print(f"Score range: {min_score:.3f} - {max_score:.3f}")
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in sorted_data],
        labels=[d['labels'] for d in sorted_data],
        parents=[d['parents'] for d in sorted_data],
        values=[d['values'] for d in sorted_data],
        branchvalues="total",
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[1]}<br>' +
                     'Weakness Level: %{customdata[2]}<br>' +
                     'Std Dev: %{customdata[3]:.3f}<br>' +
                     'Coverage: %{customdata[4]:.1%}<br>' +
                     '<extra></extra>',
        customdata=[[d['full_capability'], d['dove_score'], d['weakness_level'], 
                    d['std_score'], d['coverage']] for d in sorted_data],
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
        maxdepth=3,
        textfont=dict(size=9, family="Arial", color="black"),
        insidetextorientation='horizontal'
    ))
    
    # Academic paper styling
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:11px; color:#666'>Analysis of {len(sorted_data)} Shortened Capability Clusters</span>",
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
    
    return fig, sorted_data

def print_working_results_summary(weakness_data):
    """Print summary for working shortened sunburst"""
    print("\n" + "="*60)
    print("WORKING SHORTENED SUNBURST - RESULTS SUMMARY")
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
            
            # Show top examples
            for i, item in enumerate(items[:3], 1):
                print(f"  {i}. {item['full_capability'][:40]}... ({item['dove_score']})")

def main():
    """Main execution"""
    print("🚀 Creating Working Shortened Sunburst...")
    print("="*60)
    
    try:
        # Load shortened tree
        tree = load_shortened_tree()
        
        # Extract data using simple approach
        print("Extracting data using simple approach...")
        data = extract_simple_shortened_data(tree)
        
        if not data:
            print("❌ No data extracted!")
            return
        
        print(f"Extracted {len(data)} data points")
        
        # Create working sunburst
        print("Creating working sunburst...")
        fig, weakness_data = create_working_shortened_sunburst(data, top_n=25)
        
        if fig:
            fig.write_html("working_shortened_sunburst.html")
            print("✅ Working shortened sunburst saved to working_shortened_sunburst.html")
        
        # Create extended version
        print("Creating extended version...")
        fig_ext, weakness_data_ext = create_working_shortened_sunburst(data, 
                                                                      title="DOVE Weakness Analysis - Extended Shortened",
                                                                      top_n=40)
        if fig_ext:
            fig_ext.write_html("working_shortened_extended_sunburst.html")
            print("✅ Extended version saved to working_shortened_extended_sunburst.html")
        
        # Print results
        if weakness_data:
            print_working_results_summary(weakness_data)
        
        print("\n" + "="*60)
        print("🎉 WORKING SHORTENED SUNBURST COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  • working_shortened_sunburst.html - Main figure (25 categories)")
        print("  • working_shortened_extended_sunburst.html - Extended (40 categories)")
        print("\n📊 Features:")
        print("  • Uses the same approach as minimal working version")
        print("  • Pre-shortened capabilities from your JSON")
        print("  • Red = Low accuracy (weak), Blue = High accuracy (strong)")
        print("  • Unique IDs to prevent rendering issues")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 