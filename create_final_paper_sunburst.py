#!/usr/bin/env python3
"""
Final Paper-Quality DOVE Sunburst

Uses the working structure from improved version but with academic styling
"""

import json
import plotly.graph_objects as go
import statistics
from collections import defaultdict
import hashlib

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
                indices.extend(collect_question_indices(subtree))
    
    return indices

def create_unique_id(text, level, index):
    """Create a unique ID that won't have duplicates"""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"L{level}_{index}_{text_hash}"

def clean_paper_label(text, target_words=4):
    """Clean text for paper - keep it simple and working"""
    if not text:
        return text
    
    words = text.split()
    
    # Remove only the most common repetitive words
    remove_words = {'analyzing', 'evaluating', 'and', 'the', 'of'}
    
    # Filter words
    filtered_words = []
    for word in words:
        if word.lower() not in remove_words:
            filtered_words.append(word)
    
    # Take first target_words, but ensure we have something
    if len(filtered_words) >= target_words:
        result = ' '.join(filtered_words[:target_words])
    elif len(filtered_words) > 0:
        result = ' '.join(filtered_words)
    else:
        result = ' '.join(words[:target_words])
    
    return result.capitalize()

def extract_flat_weakness_data(node, dove_scores, level=0, max_levels=3):
    """Extract flat list of weakness data with unique IDs - using working version"""
    data = []
    node_index = 0
    
    def traverse(current_node, current_level, parent_id=""):
        nonlocal node_index
        
        if not isinstance(current_node, dict) or current_level >= max_levels:
            return
        
        capability = current_node.get('capability', f'Node_{node_index}')
        
        # Get question indices and calculate DOVE stats
        question_indices = collect_question_indices(current_node)
        subtree_scores = []
        for idx in question_indices:
            if str(idx) in dove_scores:
                subtree_scores.append(dove_scores[str(idx)])
        
        if subtree_scores and len(subtree_scores) >= 3:
            mean_score = statistics.mean(subtree_scores)
            question_count = len(subtree_scores)
            
            # Create unique ID and clean label
            unique_id = create_unique_id(capability, current_level, node_index)
            clean_label = clean_paper_label(capability)
            
            # Add data point
            data.append({
                'ids': unique_id,
                'labels': clean_label,
                'parents': parent_id,
                'values': question_count,
                'dove_score': round(mean_score, 3),
                'level': current_level,
                'full_capability': capability,
                'weakness_level': 'Critical' if mean_score < 0.3 else 
                                'High' if mean_score < 0.5 else 
                                'Moderate' if mean_score < 0.7 else 'Low'
            })
            
            # Recurse into subtrees
            if 'subtrees' in current_node and isinstance(current_node['subtrees'], list):
                for subtree in current_node['subtrees']:
                    node_index += 1
                    traverse(subtree, current_level + 1, unique_id)
        
        node_index += 1
    
    traverse(node, level)
    return data

def create_final_paper_sunburst(data, title="DOVE Robustness Weakness Analysis", top_n=35):
    """Create final paper-quality sunburst"""
    
    if not data:
        print("❌ No data available for sunburst")
        return None, None
    
    # Sort by weakness and take top N
    sorted_data = sorted(data, key=lambda x: x['dove_score'])[:top_n]
    
    print(f"Creating final paper sunburst with {len(sorted_data)} categories")
    
    # Verify no duplicate IDs
    ids = [d['ids'] for d in sorted_data]
    if len(ids) != len(set(ids)):
        print("⚠️ Warning: Duplicate IDs detected, fixing...")
        for i, item in enumerate(sorted_data):
            item['ids'] = f"node_{i}_{item['ids']}"
    
    # Create color mapping
    min_score = min(d['dove_score'] for d in sorted_data)
    max_score = max(d['dove_score'] for d in sorted_data)
    
    colors = []
    for d in sorted_data:
        if max_score > min_score:
            # Normalize: 1 = red (worst/weakest), 0 = blue (better/stronger)
            norm_score = 1 - (d['dove_score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 1  # If all same score, make them red (weak)
        colors.append(norm_score)
    
    print(f"Score range: {min_score:.3f} - {max_score:.3f}")
    print("Top 5 weakest categories:")
    for i, item in enumerate(sorted_data[:5]):
        print(f"  {i+1}. {item['labels']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in sorted_data],
        labels=[f"{d['labels']}<br>{d['dove_score']}" for d in sorted_data],
        parents=[d['parents'] for d in sorted_data],
        values=[d['values'] for d in sorted_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[0]}<br>' +
                     'Weakness Level: %{customdata[1]}<br>' +
                     'Full Description: %{customdata[2]}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['weakness_level'], 
                    d['full_capability'][:70] + "..." if len(d['full_capability']) > 70 
                    else d['full_capability']] for d in sorted_data],
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
                ticktext=["Strong", "Moderate", "Weak", "Critical"],
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
            text=f"{title}<br><span style='font-size:11px; color:#666'>Hierarchical Analysis of {len(sorted_data)} Capability Clusters</span>",
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

def print_paper_results(weakness_data):
    """Print results for paper"""
    print("\n" + "="*70)
    print("PAPER RESULTS SUMMARY")
    print("="*70)
    
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
    """Main execution function"""
    print("Creating Final Paper-Quality DOVE Sunburst...")
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract weakness data
        print("Extracting weakness data...")
        weakness_data = extract_flat_weakness_data(eval_tree, dove_scores)
        
        if not weakness_data:
            print("❌ No weakness data extracted!")
            return
        
        print(f"Extracted {len(weakness_data)} weakness categories")
        
        # Create final paper sunburst
        print("Creating final paper sunburst...")
        final_fig, sorted_data = create_final_paper_sunburst(weakness_data, top_n=35)
        if final_fig:
            final_fig.write_html("final_paper_sunburst.html")
            print("✅ Final paper sunburst saved to final_paper_sunburst.html")
        
        # Create main figure version (even cleaner)
        print("Creating main figure version...")
        main_fig, main_data = create_final_paper_sunburst(weakness_data, 
                                                         title="DOVE Weakness Profile", 
                                                         top_n=25)
        if main_fig:
            main_fig.write_html("figure1_final_sunburst.html")
            print("✅ Main figure saved to figure1_final_sunburst.html")
        
        # Print results
        if sorted_data:
            print_paper_results(sorted_data)
        
        print("\n" + "="*70)
        print("FINAL PAPER SUNBURST COMPLETE!")
        print("="*70)
        print("Files for paper:")
        print("  • final_paper_sunburst.html - Complete version (35 categories)")
        print("  • figure1_final_sunburst.html - Main figure (25 categories)")
        
        print("\n📊 These should work and display correctly!")
        print("  • Uses proven working structure")
        print("  • Academic styling and fonts")
        print("  • Clean labels without repetitive words")
        print("  • Proper color coding (Red=Critical, Blue=Strong)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 