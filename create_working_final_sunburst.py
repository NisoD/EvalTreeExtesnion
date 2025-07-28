#!/usr/bin/env python3
"""
Working Final Paper Sunburst - Based on Improved Version Logic
"""

import json
import plotly.graph_objects as go
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

def clean_capability_label(text, max_words=4):
    """Clean capability text for paper - remove common academic words"""
    if not text:
        return text
    
    words = text.split()
    
    # Remove common academic words
    remove_words = {'analyzing', 'evaluating', 'synthesizing', 'and', 'the', 'of', 'for', 'in', 'with', 'to'}
    
    # Filter words
    filtered_words = []
    for word in words:
        if word.lower() not in remove_words:
            filtered_words.append(word)
    
    # Take first max_words, but ensure we have something
    if len(filtered_words) >= max_words:
        result = ' '.join(filtered_words[:max_words])
    elif len(filtered_words) > 0:
        result = ' '.join(filtered_words)
    else:
        result = ' '.join(words[:max_words])
    
    return result.capitalize()

def extract_hierarchical_data(node, dove_scores, path="", level=0, max_levels=4):
    """Extract hierarchical data using the SAME logic as improved version"""
    data = []
    
    if not isinstance(node, dict) or level >= max_levels:
        return data
    
    capability = node.get('capability', f'Level_{level}')
    
    # Clean capability name for paper
    capability_short = clean_capability_label(capability, max_words=4)
    
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
        
        # Determine weakness level
        if mean_score < 0.3:
            weakness_level = "Critical"
        elif mean_score < 0.5:
            weakness_level = "High"
        elif mean_score < 0.7:
            weakness_level = "Moderate"
        else:
            weakness_level = "Low"
        
        # Create path for sunburst - SAME as improved version
        current_path = [p for p in path.split(" > ") if p] + [capability_short]
        
        # Add data point - SAME structure as improved version
        data.append({
            'ids': " > ".join(current_path),
            'labels': capability_short,
            'parents': " > ".join(current_path[:-1]) if len(current_path) > 1 else "",
            'values': question_count,
            'dove_score': round(mean_score, 3),
            'weakness_level': weakness_level,
            'color_value': mean_score,
            'level': level,
            'question_count': question_count,
            'full_capability': capability
        })
    
    # Recurse into subtrees - SAME as improved version
    if 'subtrees' in node and isinstance(node['subtrees'], list):
        for subtree in node['subtrees']:
            new_path = f"{path} > {capability_short}" if path else capability_short
            data.extend(extract_hierarchical_data(subtree, dove_scores, new_path, level + 1, max_levels))
    
    return data

def create_paper_sunburst(hierarchical_data, title="DOVE Weakness Profile", top_n=30):
    """Create paper-quality sunburst using working logic from improved version"""
    
    # Filter and sort - SAME as improved version
    filtered_data = [d for d in hierarchical_data if d['question_count'] >= 3]
    sorted_data = sorted(filtered_data, key=lambda x: x['dove_score'])
    
    # Take only the weakest top_n categories
    top_weak_data = sorted_data[:top_n]
    
    if not top_weak_data:
        print("❌ No data after filtering!")
        return None, None
    
    print(f"Creating sunburst with {len(top_weak_data)} categories")
    print("Top 5 weakest categories:")
    for i, item in enumerate(top_weak_data[:5]):
        print(f"  {i+1}. {item['labels']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Color mapping - continuous scale
    min_score = min(d['dove_score'] for d in top_weak_data)
    max_score = max(d['dove_score'] for d in top_weak_data)
    
    colors = []
    for d in top_weak_data:
        if max_score > min_score:
            # Normalize: 0 = red (weakest/lowest DOVE), 1 = blue (strongest/highest DOVE)
            norm_score = (d['dove_score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 0  # All same, make red (weak)
        colors.append(norm_score)
    
    # Create sunburst - using go.Sunburst like improved version
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
                     'Full Description: %{customdata[2]}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['weakness_level'], 
                    d['full_capability'][:60] + "..." if len(d['full_capability']) > 60 
                    else d['full_capability']] for d in top_weak_data],
        marker=dict(
            colorscale='RdYlBu_r',  # Red to Blue (reversed)
            cmin=0,
            cmax=1,
            colors=colors,
            colorbar=dict(
                title=dict(
                    text="Robustness<br>Level",
                    font=dict(size=12, family="Arial", color="black")
                ),
                tickmode="array",
                tickvals=[0, 0.33, 0.66, 1.0],
                ticktext=["Critical", "Weak", "Moderate", "Strong"],
                tickfont=dict(size=10, family="Arial", color="black"),
                len=0.6,
                thickness=12,
                x=1.02
            ),
            line=dict(color="white", width=1.5)
        ),
        maxdepth=4,  # Allow more depth like improved version
        textfont=dict(size=9, family="Arial", color="black"),
        insidetextorientation='horizontal'
    ))
    
    # Academic paper styling
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:11px; color:#666'>Analysis of {len(top_weak_data)} Weakest Capability Clusters</span>",
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

def print_results_summary(weakness_data):
    """Print summary for paper"""
    print("\n" + "="*60)
    print("WORKING FINAL SUNBURST - RESULTS SUMMARY")
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
                print(f"  {i}. {item['labels']} ({item['dove_score']})")

def main():
    """Main execution"""
    print("🚀 Creating Working Final Paper Sunburst...")
    print("="*60)
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract hierarchical data using improved version logic
        print("Extracting hierarchical data...")
        hierarchical_data = extract_hierarchical_data(eval_tree, dove_scores)
        
        if not hierarchical_data:
            print("❌ No hierarchical data extracted!")
            return
        
        print(f"Extracted {len(hierarchical_data)} hierarchical data points")
        
        # Create paper sunburst
        print("Creating paper sunburst...")
        fig, weakness_data = create_paper_sunburst(hierarchical_data, top_n=25)
        
        if fig:
            fig.write_html("working_final_paper_sunburst.html")
            print("✅ Working final paper sunburst saved to working_final_paper_sunburst.html")
        
        # Create extended version
        print("Creating extended version...")
        fig_ext, weakness_data_ext = create_paper_sunburst(hierarchical_data, 
                                                          title="DOVE Weakness Analysis - Extended",
                                                          top_n=40)
        if fig_ext:
            fig_ext.write_html("working_final_extended_sunburst.html")
            print("✅ Extended version saved to working_final_extended_sunburst.html")
        
        # Print results
        if weakness_data:
            print_results_summary(weakness_data)
        
        print("\n" + "="*60)
        print("🎉 WORKING FINAL SUNBURST COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  • working_final_paper_sunburst.html - Main paper figure (25 categories)")
        print("  • working_final_extended_sunburst.html - Extended version (40 categories)")
        print("\n📊 These use the EXACT same logic as the working improved version!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 