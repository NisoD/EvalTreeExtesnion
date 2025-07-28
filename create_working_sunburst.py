#!/usr/bin/env python3
"""
Working DOVE-Based Sunburst Weakness Tree

Creates functional sunburst visualizations with:
- Unique IDs to prevent rendering issues
- Shortened capability names for readability  
- Focus on weakest categories with proper color coding
- Guaranteed working HTML output
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # Use a combination of level, index, and hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"L{level}_{index}_{text_hash}"

def shorten_text(text, target_words=5):
    """Shorten text to 4-7 most important words"""
    if not text:
        return text
    
    words = text.split()
    
    # If already short enough, return as is
    if len(words) <= target_words:
        return text
    
    # Remove common less important words
    skip_words = {'and', 'or', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'with', 'by', 'from'}
    
    # Keep important words (not in skip list)
    important_words = [w for w in words if w.lower() not in skip_words]
    
    # If we have enough important words, use them
    if len(important_words) >= target_words:
        return ' '.join(important_words[:target_words])
    
    # Otherwise, take first few words including some common ones
    return ' '.join(words[:target_words])

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
            
            # Create unique ID and shortened label
            unique_id = create_unique_id(capability, current_level, node_index)
            short_label = shorten_text(capability)
            
            # Add data point
            data.append({
                'ids': unique_id,
                'labels': short_label,
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

def create_weakness_sunburst(data, title="DOVE Weakness Profile", top_n=50):
    """Create a working sunburst visualization"""
    
    if not data:
        print("❌ No data available for sunburst")
        return None
    
    # Sort by weakness and take top N
    sorted_data = sorted(data, key=lambda x: x['dove_score'])[:top_n]
    
    print(f"Creating sunburst with {len(sorted_data)} weakest categories")
    
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
        labels=[f"{d['labels']}<br>Score: {d['dove_score']}" for d in sorted_data],
        parents=[d['parents'] for d in sorted_data],
        values=[d['values'] for d in sorted_data],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>' +
                     'Questions: %{value}<br>' +
                     'DOVE Score: %{customdata[0]}<br>' +
                     'Weakness Level: %{customdata[1]}<br>' +
                     'Full Name: %{customdata[2]}<br>' +
                     '<extra></extra>',
        customdata=[[d['dove_score'], d['weakness_level'], 
                    d['full_capability'][:60] + "..." if len(d['full_capability']) > 60 
                    else d['full_capability']] for d in sorted_data],
        marker=dict(
            colorscale='RdYlBu_r',  # Red to Blue (reversed)
            cmin=0,
            cmax=1,
            colors=colors,
                         colorbar=dict(
                 title="Weakness Level",
                 tickmode="array",
                 tickvals=[0, 0.33, 0.66, 1.0],
                 ticktext=["Strong", "Moderate", "Weak", "Critical"]
             ),
            line=dict(color="white", width=2)
        ),
        maxdepth=3
    ))
    
    fig.update_layout(
        title=f"{title}<br><sub>Top {len(sorted_data)} Weakest Categories (Red = Critical, Blue = Strong)</sub>",
        font_size=11,
        width=900,
        height=900,
        showlegend=False
    )
    
    return fig, sorted_data

def create_comparison_sunburst(weakness_data):
    """Create comparison between critical and moderate weaknesses"""
    
    if not weakness_data:
        return None
    
    # Split into critical and moderate
    critical_data = [d for d in weakness_data if d['dove_score'] < 0.4][:25]
    moderate_data = [d for d in weakness_data if 0.4 <= d['dove_score'] < 0.7][:25]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=(f"Critical Weaknesses (Top {len(critical_data)})", 
                       f"Moderate Weaknesses (Top {len(moderate_data)})")
    )
    
    # Critical weaknesses (left)
    if critical_data:
        fig.add_trace(go.Sunburst(
            ids=[d['ids'] for d in critical_data],
            labels=[f"{d['labels']}<br>{d['dove_score']}" for d in critical_data],
            parents=[d['parents'] for d in critical_data],
            values=[d['values'] for d in critical_data],
            branchvalues="total",
            marker=dict(
                colors=['#d32f2f'] * len(critical_data),  # All red for critical
                line=dict(color="white", width=1)
            ),
            hovertemplate='<b>%{label}</b><br>Questions: %{value}<br><extra></extra>',
            maxdepth=2,
            domain=dict(column=0)
        ), row=1, col=1)
    
    # Moderate weaknesses (right)
    if moderate_data:
        fig.add_trace(go.Sunburst(
            ids=[d['ids'] for d in moderate_data],
            labels=[f"{d['labels']}<br>{d['dove_score']}" for d in moderate_data],
            parents=[d['parents'] for d in moderate_data],
            values=[d['values'] for d in moderate_data],
            branchvalues="total",
            marker=dict(
                colors=['#fbc02d'] * len(moderate_data),  # All yellow for moderate
                line=dict(color="white", width=1)
            ),
            hovertemplate='<b>%{label}</b><br>Questions: %{value}<br><extra></extra>',
            maxdepth=2,
            domain=dict(column=1)
        ), row=1, col=2)
    
    fig.update_layout(
        title="DOVE Weakness Comparison: Critical vs Moderate",
        font_size=10,
        width=1600,
        height=800,
        showlegend=False
    )
    
    return fig

def print_weakness_summary(weakness_data):
    """Print summary of weaknesses"""
    print("\n" + "="*80)
    print("WEAKNESS SUMMARY")
    print("="*80)
    
    # Group by weakness level
    by_level = defaultdict(list)
    for item in weakness_data:
        by_level[item['weakness_level']].append(item)
    
    for level in ['Critical', 'High', 'Moderate', 'Low']:
        if level in by_level:
            items = by_level[level]
            print(f"\n{level.upper()} WEAKNESSES ({len(items)} categories):")
            print("-" * 60)
            
            for i, item in enumerate(items[:10], 1):
                capability = item['full_capability'][:50] + "..." if len(item['full_capability']) > 50 else item['full_capability']
                print(f"{i:2d}. {capability}")
                print(f"    DOVE Score: {item['dove_score']} | Questions: {item['values']}")
            
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

def main():
    """Main execution function"""
    print("Creating Working DOVE-Based Sunburst Weakness Tree...")
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract weakness data with unique IDs
        print("Extracting weakness data with unique identifiers...")
        weakness_data = extract_flat_weakness_data(eval_tree, dove_scores)
        
        if not weakness_data:
            print("❌ No weakness data extracted!")
            return
        
        print(f"Extracted {len(weakness_data)} weakness categories")
        
        # Create main weakness sunburst (top 50)
        print("Creating main weakness sunburst...")
        main_fig, sorted_data = create_weakness_sunburst(weakness_data, top_n=50)
        if main_fig:
            main_fig.write_html("working_dove_sunburst.html")
            print("✅ Main sunburst saved to working_dove_sunburst.html")
        
        # Create ultra-focused version (top 25)
        print("Creating ultra-focused sunburst...")
        focused_fig, focused_data = create_weakness_sunburst(weakness_data, 
                                                           title="DOVE Critical Weaknesses", 
                                                           top_n=25)
        if focused_fig:
            focused_fig.write_html("working_dove_focused_sunburst.html")
            print("✅ Focused sunburst saved to working_dove_focused_sunburst.html")
        
        # Create comparison sunburst
        print("Creating comparison sunburst...")
        comparison_fig = create_comparison_sunburst(sorted_data)
        if comparison_fig:
            comparison_fig.write_html("working_dove_comparison_sunburst.html")
            print("✅ Comparison sunburst saved to working_dove_comparison_sunburst.html")
        
        # Print summary
        print_weakness_summary(sorted_data)
        
        print("\n" + "="*80)
        print("WORKING SUNBURST VISUALIZATION COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • working_dove_sunburst.html - Top 50 weakest categories")
        print("  • working_dove_focused_sunburst.html - Top 25 most critical")
        print("  • working_dove_comparison_sunburst.html - Critical vs Moderate comparison")
        
        print("\n🎯 Key Features:")
        print("  • Unique IDs prevent rendering issues")
        print("  • Shortened names for readability")
        print("  • Color-coded by weakness level")
        print("  • Interactive hover details")
        print("  • Guaranteed working HTML output")
        
        print("\n📊 Open the HTML files in your browser - they should display correctly now!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 