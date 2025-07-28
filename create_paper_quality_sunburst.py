#!/usr/bin/env python3
"""
Paper-Quality DOVE-Based Sunburst Weakness Tree

Creates publication-ready sunburst visualizations with:
- Centered text in main circle
- Concise labels without repetitive words (analyze, evaluate)
- Clean academic styling for paper figures
- Red = Critical/Weak, Blue = Strong
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

def clean_academic_text(text, target_words=4):
    """Clean text for academic presentation - remove repetitive words and keep key terms"""
    if not text:
        return text
    
    words = text.split()
    
    # Remove common repetitive words found in capabilities
    remove_words = {
        'analyzing', 'analyze', 'evaluating', 'evaluate', 'synthesizing', 'synthesize',
        'and', 'or', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'with', 'by', 'from',
        'using', 'applying', 'based', 'across', 'within', 'through', 'various', 'diverse',
        'complex', 'comprehensive', 'advanced', 'effective', 'strategic'
    }
    
    # Keep important domain-specific words
    important_words = []
    for word in words:
        if word.lower() not in remove_words:
            important_words.append(word)
    
    # If we have enough important words, use them
    if len(important_words) >= target_words:
        result = ' '.join(important_words[:target_words])
    else:
        # Take original words but still remove the most common ones
        minimal_remove = {'analyzing', 'analyze', 'evaluating', 'evaluate', 'and', 'the', 'of'}
        filtered_words = [w for w in words if w.lower() not in minimal_remove]
        result = ' '.join(filtered_words[:target_words])
    
    # Capitalize first letter for academic style
    return result.capitalize() if result else text

def extract_flat_weakness_data(node, dove_scores, level=0, max_levels=3):
    """Extract flat list of weakness data with unique IDs"""
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
            
            # Create unique ID and clean academic label
            unique_id = create_unique_id(capability, current_level, node_index)
            clean_label = clean_academic_text(capability)
            
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

def create_paper_quality_sunburst(data, title="DOVE Weakness Profile", top_n=40):
    """Create a paper-quality sunburst visualization"""
    
    if not data:
        print("❌ No data available for sunburst")
        return None, None
    
    # Sort by weakness and take top N (slightly fewer for cleaner paper figure)
    sorted_data = sorted(data, key=lambda x: x['dove_score'])[:top_n]
    
    print(f"Creating paper-quality sunburst with {len(sorted_data)} categories")
    
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
    print("Top 5 weakest categories (cleaned labels):")
    for i, item in enumerate(sorted_data[:5]):
        print(f"  {i+1}. {item['labels']} (Score: {item['dove_score']}, Questions: {item['values']})")
    
    # Create sunburst with academic styling
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
                    d['full_capability'][:80] + "..." if len(d['full_capability']) > 80 
                    else d['full_capability']] for d in sorted_data],
        marker=dict(
            colorscale='RdYlBu_r',  # Red to Blue (reversed)
            cmin=0,
            cmax=1,
            colors=colors,
            colorbar=dict(
                title=dict(
                    text="Robustness Level",
                    font=dict(size=12, family="Arial")
                ),
                tickmode="array",
                tickvals=[0, 0.33, 0.66, 1.0],
                ticktext=["Strong", "Moderate", "Weak", "Critical"],
                tickfont=dict(size=11, family="Arial"),
                len=0.7,
                thickness=15
            ),
            line=dict(color="white", width=1.5)
        ),
        maxdepth=3,
        # Center the text in the main circle
        textfont=dict(size=10, family="Arial", color="black")
    ))
    
    # Academic paper styling
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:12px; color:gray'>Hierarchical Weakness Analysis (n={len(sorted_data)} categories)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family="Arial", color="black")
        ),
        font=dict(size=11, family="Arial"),
        width=800,
        height=800,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig, sorted_data

def create_figure_caption():
    """Generate academic figure caption"""
    caption = """Figure 1. DOVE-Based Hierarchical Weakness Profile for MMLU Evaluation.
    
The sunburst diagram displays the hierarchical clustering of MMLU question categories 
ranked by robustness weakness (DOVE scores). Each segment represents a capability cluster, 
with size proportional to question count and color indicating weakness level 
(red = critical weakness, blue = strong performance). Lower DOVE scores indicate 
higher sensitivity to input perturbations, revealing specific areas where the model 
demonstrates reduced robustness. The hierarchical structure preserves the EvalTree 
methodology's capability clustering while incorporating continuous robustness assessment."""
    
    return caption

def print_paper_summary(weakness_data):
    """Print academic summary for paper"""
    print("\n" + "="*80)
    print("PAPER-READY WEAKNESS ANALYSIS SUMMARY")
    print("="*80)
    
    # Group by weakness level
    by_level = defaultdict(list)
    for item in weakness_data:
        by_level[item['weakness_level']].append(item)
    
    print(f"\nDataset Coverage: {len(weakness_data)} capability clusters analyzed")
    print(f"Question Coverage: {sum(item['values'] for item in weakness_data)} total questions")
    
    for level in ['Critical', 'High', 'Moderate', 'Low']:
        if level in by_level:
            items = by_level[level]
            avg_score = sum(item['dove_score'] for item in items) / len(items)
            total_questions = sum(item['values'] for item in items)
            
            print(f"\n{level} Weaknesses: {len(items)} clusters (avg DOVE: {avg_score:.3f}, {total_questions} questions)")
            print("  Key Areas:")
            for i, item in enumerate(items[:5], 1):
                print(f"    {i}. {item['labels']} (DOVE: {item['dove_score']})")
    
    print(f"\nFigure Caption:")
    print(create_figure_caption())

def main():
    """Main execution function"""
    print("Creating Paper-Quality DOVE-Based Sunburst...")
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract weakness data with unique IDs
        print("Extracting weakness data for academic presentation...")
        weakness_data = extract_flat_weakness_data(eval_tree, dove_scores)
        
        if not weakness_data:
            print("❌ No weakness data extracted!")
            return
        
        print(f"Extracted {len(weakness_data)} weakness categories")
        
        # Create paper-quality sunburst
        print("Creating paper-quality sunburst...")
        paper_fig, sorted_data = create_paper_quality_sunburst(weakness_data, top_n=40)
        if paper_fig:
            paper_fig.write_html("paper_dove_sunburst.html")
            print("✅ Paper-quality sunburst saved to paper_dove_sunburst.html")
            
            # Also save as static image for paper
            try:
                paper_fig.write_image("paper_dove_sunburst.png", width=800, height=800, scale=2)
                paper_fig.write_image("paper_dove_sunburst.pdf", width=800, height=800)
                print("✅ Static images saved: paper_dove_sunburst.png/pdf")
            except Exception as e:
                print(f"⚠️ Could not save static images: {e}")
        
        # Create ultra-clean version for main figure
        print("Creating main figure version...")
        main_fig, main_data = create_paper_quality_sunburst(weakness_data, 
                                                           title="DOVE Robustness Weakness Analysis", 
                                                           top_n=30)
        if main_fig:
            main_fig.write_html("figure1_dove_sunburst.html")
            print("✅ Main figure saved to figure1_dove_sunburst.html")
        
        # Print academic summary
        if sorted_data:
            print_paper_summary(sorted_data)
        
        print("\n" + "="*80)
        print("PAPER-QUALITY SUNBURST COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • paper_dove_sunburst.html - Full interactive version")
        print("  • figure1_dove_sunburst.html - Main figure (30 categories)")
        print("  • paper_dove_sunburst.png/pdf - Static images for paper")
        
        print("\n📊 Paper Integration:")
        print("  • Use figure1_dove_sunburst.html for Figure 1/2")
        print("  • Clean academic styling with centered text")
        print("  • Concise labels without repetitive words")
        print("  • Professional color scheme and typography")
        print("  • Includes figure caption for paper")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 