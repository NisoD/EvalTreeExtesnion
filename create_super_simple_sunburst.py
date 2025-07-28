#!/usr/bin/env python3
"""
Super Simple Working Sunburst
Based on the minimal working approach that we know worked before
"""

import json
import plotly.graph_objects as go

def load_shortened_tree():
    """Load the shortened tree JSON"""
    print("Loading shortened tree...")
    
    try:
        with open('shortened tree.json', 'r') as f:
            content = f.read()
            tree = json.loads(content)
    except json.JSONDecodeError:
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
    
    return tree

def extract_super_simple_data(tree):
    """Extract data using the EXACT same approach as minimal working version"""
    print("Extracting data using super simple approach...")
    data = []
    
    # Only look at top level subtrees, no hierarchy
    if 'subtrees' in tree and isinstance(tree['subtrees'], list):
        for i, subtree in enumerate(tree['subtrees'][:10]):  # Just first 10
            if isinstance(subtree, dict) and 'capability' in subtree:
                capability = subtree['capability']
                weakness_stats = subtree.get('weakness_stats', {})
                
                if weakness_stats and weakness_stats.get('question_count', 0) >= 10:  # Higher threshold
                    mean_score = weakness_stats.get('mean_dove_score', 0.5)
                    question_count = weakness_stats.get('question_count', 0)
                    
                    # Super simple label
                    label = capability[:25] + "..." if len(capability) > 25 else capability
                    
                    data.append({
                        'ids': f"item_{i}",
                        'labels': f"{label}<br>{mean_score:.3f}",
                        'parents': "",  # All top level
                        'values': question_count,
                        'score': mean_score
                    })
                    
                    print(f"Added: {label} (Score: {mean_score:.3f}, Questions: {question_count})")
    
    return data

def create_super_simple_sunburst(data):
    """Create super simple sunburst"""
    
    if not data:
        print("❌ No data for sunburst")
        return None
    
    print(f"Creating super simple sunburst with {len(data)} items")
    
    # Sort by weakness (lowest score first)
    sorted_data = sorted(data, key=lambda x: x['score'])
    
    # Color mapping - simple approach
    min_score = min(d['score'] for d in sorted_data)
    max_score = max(d['score'] for d in sorted_data)
    
    colors = []
    for d in sorted_data:
        if max_score > min_score:
            # 0 = red (weak), 1 = blue (strong)
            norm_score = (d['score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 0.5
        colors.append(norm_score)
    
    print(f"Score range: {min_score:.3f} - {max_score:.3f}")
    print("Items to display:")
    for i, item in enumerate(sorted_data):
        print(f"  {i+1}. {item['labels']} (Value: {item['values']})")
    
    # Create sunburst - EXACT same as minimal working version
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in sorted_data],
        labels=[d['labels'] for d in sorted_data],
        parents=[d['parents'] for d in sorted_data],
        values=[d['values'] for d in sorted_data],
        marker=dict(
            colorscale='RdYlBu_r',
            colors=colors,
            colorbar=dict(title="Weakness Level")
        )
    ))
    
    fig.update_layout(
        title="Super Simple DOVE Sunburst",
        width=600,
        height=600
    )
    
    return fig

def main():
    print("🚀 Creating Super Simple Working Sunburst...")
    print("="*50)
    
    try:
        # Load tree
        tree = load_shortened_tree()
        
        # Extract simple data
        data = extract_super_simple_data(tree)
        
        if not data:
            print("❌ No data extracted!")
            return
        
        # Create sunburst
        fig = create_super_simple_sunburst(data)
        
        if fig:
            fig.write_html("super_simple_sunburst.html")
            print("✅ Super simple sunburst saved to super_simple_sunburst.html")
            
            # Also create a version with different name
            fig.write_html("final_working_sunburst.html")
            print("✅ Also saved as final_working_sunburst.html")
        
        print("\n" + "="*50)
        print("🎉 SUPER SIMPLE SUNBURST COMPLETE!")
        print("="*50)
        print("Try opening:")
        print("  • super_simple_sunburst.html")
        print("  • final_working_sunburst.html")
        print("\nThis uses the EXACT same approach as the minimal version")
        print("that worked before, but with your shortened data.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 