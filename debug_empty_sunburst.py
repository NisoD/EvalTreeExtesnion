#!/usr/bin/env python3
"""
Debug Empty Sunburst Issue
"""

import json
import plotly.graph_objects as go
import statistics
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

def create_simple_test_sunburst():
    """Create a super simple test sunburst to verify Plotly works"""
    print("Creating simple test sunburst...")
    
    # Simple test data
    fig = go.Figure(go.Sunburst(
        ids=["A", "B", "C", "A.1", "A.2", "B.1"],
        labels=["A", "B", "C", "A1", "A2", "B1"],
        parents=["", "", "", "A", "A", "B"],
        values=[10, 15, 20, 5, 5, 15],
    ))
    
    fig.update_layout(title="Simple Test Sunburst")
    fig.write_html("simple_test_sunburst.html")
    print("✅ Simple test saved to simple_test_sunburst.html")

def debug_data_extraction():
    """Debug the data extraction process step by step"""
    print("\n" + "="*50)
    print("DEBUGGING DATA EXTRACTION")
    print("="*50)
    
    dove_scores, eval_tree = load_data()
    
    print(f"\n1. Tree root structure:")
    print(f"   Keys: {list(eval_tree.keys())}")
    print(f"   Capability: {eval_tree.get('capability', 'None')[:100]}...")
    print(f"   Size: {eval_tree.get('size')}")
    print(f"   Depth: {eval_tree.get('depth')}")
    
    # Check first level subtrees
    if 'subtrees' in eval_tree:
        subtrees = eval_tree['subtrees']
        print(f"\n2. Subtrees type: {type(subtrees)}")
        if isinstance(subtrees, list):
            print(f"   Number of subtrees: {len(subtrees)}")
            if len(subtrees) > 0:
                first_subtree = subtrees[0]
                print(f"   First subtree keys: {list(first_subtree.keys()) if isinstance(first_subtree, dict) else 'Not dict'}")
                if isinstance(first_subtree, dict):
                    print(f"   First subtree capability: {first_subtree.get('capability', 'None')[:50]}...")
        elif isinstance(subtrees, int):
            print(f"   Subtrees is integer: {subtrees}")
    
    # Test question collection on root
    print(f"\n3. Testing question collection on root:")
    root_questions = collect_question_indices(eval_tree)
    print(f"   Found {len(root_questions)} question indices")
    if len(root_questions) > 0:
        print(f"   First 5: {root_questions[:5]}")
        
        # Check DOVE score matches
        dove_matches = 0
        for idx in root_questions[:10]:
            if str(idx) in dove_scores:
                dove_matches += 1
        print(f"   DOVE matches in first 10: {dove_matches}/10")
    
    # Test on first subtree if exists
    if 'subtrees' in eval_tree and isinstance(eval_tree['subtrees'], list) and len(eval_tree['subtrees']) > 0:
        first_subtree = eval_tree['subtrees'][0]
        if isinstance(first_subtree, dict):
            print(f"\n4. Testing on first subtree:")
            subtree_questions = collect_question_indices(first_subtree)
            print(f"   Found {len(subtree_questions)} question indices")
            if len(subtree_questions) > 0:
                print(f"   First 5: {subtree_questions[:5]}")
                
                # Check DOVE matches
                dove_matches = 0
                dove_scores_found = []
                for idx in subtree_questions[:10]:
                    if str(idx) in dove_scores:
                        dove_matches += 1
                        dove_scores_found.append(dove_scores[str(idx)])
                
                print(f"   DOVE matches in first 10: {dove_matches}/10")
                if dove_scores_found:
                    mean_score = statistics.mean(dove_scores_found)
                    print(f"   Mean DOVE score: {mean_score:.3f}")
    
    return dove_scores, eval_tree

def create_minimal_working_sunburst(dove_scores, eval_tree):
    """Create minimal sunburst with just the data we know works"""
    print("\n" + "="*50)
    print("CREATING MINIMAL WORKING SUNBURST")
    print("="*50)
    
    data = []
    
    # Try to extract just a few working data points
    if 'subtrees' in eval_tree and isinstance(eval_tree['subtrees'], list):
        for i, subtree in enumerate(eval_tree['subtrees'][:5]):  # Just first 5
            if isinstance(subtree, dict) and 'capability' in subtree:
                capability = subtree['capability']
                questions = collect_question_indices(subtree)
                
                if len(questions) >= 3:  # Need at least 3 questions
                    scores = []
                    for q in questions:
                        if str(q) in dove_scores:
                            scores.append(dove_scores[str(q)])
                    
                    if len(scores) >= 3:
                        mean_score = statistics.mean(scores)
                        
                        # Simple label
                        label = capability[:30] + "..." if len(capability) > 30 else capability
                        
                        data.append({
                            'ids': f"node_{i}",
                            'labels': f"{label}<br>{mean_score:.3f}",
                            'parents': "",
                            'values': len(scores),
                            'score': mean_score
                        })
                        
                        print(f"Added: {label} (Score: {mean_score:.3f}, Questions: {len(scores)})")
    
    print(f"\nExtracted {len(data)} data points")
    
    if len(data) == 0:
        print("❌ No data extracted!")
        return None
    
    # Create simple sunburst
    min_score = min(d['score'] for d in data)
    max_score = max(d['score'] for d in data)
    
    colors = []
    for d in data:
        if max_score > min_score:
            norm_score = 1 - (d['score'] - min_score) / (max_score - min_score)
        else:
            norm_score = 1
        colors.append(norm_score)
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in data],
        labels=[d['labels'] for d in data],
        parents=[d['parents'] for d in data],
        values=[d['values'] for d in data],
        marker=dict(
            colorscale='RdYlBu_r',
            colors=colors,
            colorbar=dict(title="Weakness Level")
        )
    ))
    
    fig.update_layout(
        title="Minimal Working DOVE Sunburst",
        width=600,
        height=600
    )
    
    fig.write_html("minimal_working_sunburst.html")
    print("✅ Minimal working sunburst saved to minimal_working_sunburst.html")
    
    return fig

def main():
    print("🔍 DEBUGGING EMPTY SUNBURST ISSUE")
    print("="*60)
    
    # Test 1: Simple test sunburst
    create_simple_test_sunburst()
    
    # Test 2: Debug data extraction
    dove_scores, eval_tree = debug_data_extraction()
    
    # Test 3: Create minimal working version
    create_minimal_working_sunburst(dove_scores, eval_tree)
    
    print("\n" + "="*60)
    print("🔍 DEBUG COMPLETE!")
    print("="*60)
    print("Check these files:")
    print("  • simple_test_sunburst.html - Basic Plotly test")
    print("  • minimal_working_sunburst.html - Minimal DOVE version")
    print("\nIf simple_test works but minimal doesn't, the issue is in data extraction.")
    print("If neither works, the issue is with Plotly/browser.")

if __name__ == "__main__":
    main() 