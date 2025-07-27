#!/usr/bin/env python3
"""
Debug script to understand the tree structure
"""

import json

def load_data():
    try:
        # Load DOVE scores
        with open('MMLU_DOVE.json', 'r') as f:
            dove_scores = json.load(f)
        
        # Load EvalTree structure
        with open('MMLU.json', 'r') as f:
            content = f.read()
            try:
                tree_data = json.loads(content)
            except json.JSONDecodeError:
                brace_count = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            tree_data = json.loads(content[:i+1])
                            break
        
        return dove_scores, tree_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def explore_tree_structure(node, depth=0, max_depth=3):
    """Explore the tree structure to understand the format"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    
    if isinstance(node, dict):
        print(f"{indent}Dict with keys: {list(node.keys())}")
        
        if 'capability' in node:
            print(f"{indent}  Capability: {node['capability'][:50]}...")
        
        if 'ranking' in node:
            ranking = node['ranking']
            if isinstance(ranking, list):
                print(f"{indent}  Ranking (leaf): {len(ranking)} items, sample: {ranking[:5]}")
            else:
                print(f"{indent}  Ranking: {type(ranking)} - {ranking}")
        
        if 'subtrees' in node:
            subtrees = node['subtrees']
            if isinstance(subtrees, list):
                print(f"{indent}  Subtrees: {len(subtrees)} children")
                for i, subtree in enumerate(subtrees[:3]):  # Only show first 3
                    print(f"{indent}    Child {i}:")
                    explore_tree_structure(subtree, depth + 2, max_depth)
            else:
                print(f"{indent}  Subtrees: {type(subtrees)} - {subtrees}")
        
        if 'size' in node:
            print(f"{indent}  Size: {node['size']}")
            
    elif isinstance(node, list):
        print(f"{indent}List with {len(node)} items")
        if node:
            print(f"{indent}  First item type: {type(node[0])}")
            if len(node) > 0:
                explore_tree_structure(node[0], depth + 1, max_depth)
    else:
        print(f"{indent}{type(node)}: {node}")

def main():
    print("Loading data...")
    dove_scores, tree_data = load_data()
    
    if not dove_scores or not tree_data:
        print("Failed to load data")
        return
    
    print(f"DOVE scores: {len(dove_scores)} entries")
    print(f"Sample DOVE keys: {list(dove_scores.keys())[:10]}")
    print(f"Sample DOVE values: {list(dove_scores.values())[:5]}")
    
    print("\nTree structure:")
    explore_tree_structure(tree_data, max_depth=4)
    
    # Try to find any leaf nodes manually
    print("\nSearching for leaf nodes...")
    
    def find_leaves(node, path="root"):
        leaves = []
        if isinstance(node, dict):
            if 'ranking' in node and isinstance(node['ranking'], list):
                leaves.append({
                    'path': path,
                    'capability': node.get('capability', 'Unknown'),
                    'ranking_size': len(node['ranking']),
                    'sample_indices': node['ranking'][:5]
                })
            elif 'subtrees' in node and isinstance(node['subtrees'], list):
                for i, subtree in enumerate(node['subtrees']):
                    leaves.extend(find_leaves(subtree, f"{path}/subtree_{i}"))
        return leaves
    
    leaves = find_leaves(tree_data)
    print(f"Found {len(leaves)} leaf nodes")
    
    for i, leaf in enumerate(leaves[:5]):  # Show first 5
        print(f"  Leaf {i+1}: {leaf['capability'][:30]}... - {leaf['ranking_size']} questions")
        print(f"    Sample indices: {leaf['sample_indices']}")
        
        # Check how many of these indices have DOVE scores
        available = sum(1 for idx in leaf['sample_indices'] if str(idx) in dove_scores)
        print(f"    DOVE coverage: {available}/{len(leaf['sample_indices'])} ({available/len(leaf['sample_indices'])*100:.1f}%)")

if __name__ == "__main__":
    main()