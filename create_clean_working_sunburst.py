#!/usr/bin/env python3
"""
Clean Working Sunburst - Using Original MMLU.json and MMLU_DOVE.json
With proper capability shortening and correct color mapping
"""

import json
import plotly.graph_objects as go
import statistics
import hashlib
from collections import defaultdict

def load_original_data():
    """Load original DOVE scores and MMLU tree structure"""
    print("Loading original data files...")
    
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

def create_unique_id(text, level, index):
    """Create a unique ID that won't have duplicates"""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"L{level}_{index}_{text_hash}"

def clean_capability_name(text, max_words=4):
    """Clean capability text - remove common academic words and keep key terms"""
    if not text:
        return text
    
    words = text.split()
    
    # Remove common academic words
    remove_words = {
        'analyzing', 'evaluating', 'synthesizing', 'and', 'the', 'of', 'for', 
        'in', 'with', 'to', 'using', 'applying', 'understanding', 'complex'
    }
    
    # Filter words but keep domain-specific terms
    filtered_words = []
    for word in words:
        if word.lower() not in remove_words:
            filtered_words.append(word)
    
    # Take first max_words, but ensure we have something meaningful
    if len(filtered_words) >= max_words:
        result = ' '.join(filtered_words[:max_words])
    elif len(filtered_words) > 0:
        result = ' '.join(filtered_words)
    else:
        # Fallback to original words if all were filtered
        result = ' '.join(words[:max_words])
    
    return result.capitalize()

def extract_clean_data(tree, dove_scores):
    """Extract data using simple approach with clean capability names"""
    print("Extracting clean data...")
    data = []
    node_index = 0
    
    def traverse(node, level=0, parent_id=""):
        nonlocal node_index
        
        if not isinstance(node, dict) or level > 3:  # Limit depth
            return
        
        capability = node.get('capability', f'Node_{node_index}')
        
        # Get question indices and calculate DOVE stats
        question_indices = collect_question_indices(node)
        subtree_scores = []
        for idx in question_indices:
            if str(idx) in dove_scores:
                subtree_scores.append(dove_scores[str(idx)])
        
        if len(subtree_scores) >= 3:  # Need at least 3 questions
            mean_score = statistics.mean(subtree_scores)
            question_count = len(subtree_scores)
            
            # Clean capability name
            clean_name = clean_capability_name(capability, max_words=4)
            
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
            
            # Add data point
            data.append({
                'ids': unique_id,
                'labels': f"{clean_name}<br>{mean_score:.3f}",
                'parents': parent_id,
                'values': question_count,
                'dove_score': round(mean_score, 3),
                'weakness_level': weakness_level,
                'level': level,
                'question_count': question_count,
                'full_capability': capability,
                'clean_name': clean_name
            })
            
            print(f"Added: {clean_name} (Score: {mean_score:.3f}, Questions: {question_count})")
            
            # Recurse into subtrees with this node as parent
            if 'subtrees' in node and isinstance(node['subtrees'], list):
                for subtree in node['subtrees']:
                    node_index += 1
                    traverse(subtree, level + 1, unique_id)
        
        node_index += 1
    
    # Start traversal
    traverse(tree)
    return data

def create_clean_working_sunburst(data, title="DOVE Weakness Profile - Clean", top_n=25):
    """Create clean working sunburst"""
    
    if not data:
        print("❌ No data available for sunburst")
        return None, None
    
    # Sort by weakness and take top N
    sorted_data = sorted(data, key=lambda x: x['dove_score'])[:top_n]
    
    print(f"Creating clean sunburst with {len(sorted_data)} categories")
    print("Top 5 weakest categories:")
    for i, item in enumerate(sorted_data[:5]):
        print(f"  {i+1}. {item['clean_name']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Verify no duplicate IDs
    ids = [d['ids'] for d in sorted_data]
    if len(ids) != len(set(ids)):
        print("⚠️ Warning: Duplicate IDs detected, fixing...")
        for i, item in enumerate(sorted_data):
            item['ids'] = f"clean_{i}_{item['ids']}"
    
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
                     'Full Description: %{customdata[3]}<br>' +
                     '<extra></extra>',
        customdata=[[d['clean_name'], d['dove_score'], d['weakness_level'], 
                    d['full_capability'][:60] + "..." if len(d['full_capability']) > 60 
                    else d['full_capability']] for d in sorted_data],
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
            text=f"{title}<br><span style='font-size:11px; color:#666'>Analysis of {len(sorted_data)} Clean Capability Clusters</span>",
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

def print_clean_results_summary(weakness_data):
    """Print summary for clean sunburst"""
    print("\n" + "="*60)
    print("CLEAN WORKING SUNBURST - RESULTS SUMMARY")
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
                print(f"  {i}. {item['clean_name']} ({item['dove_score']})")

def main():
    """Main execution"""
    print("🚀 Creating Clean Working Sunburst from Original Data...")
    print("="*60)
    
    try:
        # Load original data
        dove_scores, eval_tree = load_original_data()
        
        # Extract clean data
        print("Extracting clean data...")
        data = extract_clean_data(eval_tree, dove_scores)
        
        if not data:
            print("❌ No data extracted!")
            return
        
        print(f"Extracted {len(data)} data points")
        
        # Create clean sunburst
        print("Creating clean sunburst...")
        fig, weakness_data = create_clean_working_sunburst(data, top_n=25)
        
        if fig:
            fig.write_html("clean_working_sunburst.html")
            print("✅ Clean working sunburst saved to clean_working_sunburst.html")
        
        # Create extended version
        print("Creating extended version...")
        fig_ext, weakness_data_ext = create_clean_working_sunburst(data, 
                                                                  title="DOVE Weakness Analysis - Clean Extended",
                                                                  top_n=40)
        if fig_ext:
            fig_ext.write_html("clean_working_extended_sunburst.html")
            print("✅ Extended version saved to clean_working_extended_sunburst.html")
        
        # Print results
        if weakness_data:
            print_clean_results_summary(weakness_data)
        
        print("\n" + "="*60)
        print("🎉 CLEAN WORKING SUNBURST COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  • clean_working_sunburst.html - Main figure (25 categories)")
        print("  • clean_working_extended_sunburst.html - Extended (40 categories)")
        print("\n📊 Features:")
        print("  • Uses original clean MMLU.json and MMLU_DOVE.json")
        print("  • Proper capability name cleaning (4-word max)")
        print("  • Red = Low accuracy (weak), Blue = High accuracy (strong)")
        print("  • Unique IDs to prevent rendering issues")
        print("  • Academic styling for paper")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 