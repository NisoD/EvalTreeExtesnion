#!/usr/bin/env python3
"""
Script to combine MMLU.json, MMLU_DOVE.json, and mmlu_question_subject.json
into a new tree structure that includes DOVE scores and removes missing values.
"""

import json
import sys
from typing import Dict, Any, List, Set
import re

def extract_leaf_nodes(node: Dict[Any, Any], path: List[str] = None) -> List[Dict[Any, Any]]:
    """Extract leaf nodes (nodes with 'input' field) from the tree structure."""
    if path is None:
        path = []
    
    leaves = []
    
    # Check if this is a leaf node (has 'input' field and subtrees is not an array)
    if 'input' in node and isinstance(node.get('subtrees'), (int, type(None))):
        leaf_copy = node.copy()
        leaf_copy['tree_path'] = path.copy()
        leaves.append(leaf_copy)
    elif 'subtrees' in node and isinstance(node['subtrees'], list):
        # Recurse into subtrees
        for i, subtree in enumerate(node['subtrees']):
            new_path = path + [str(i)]
            leaves.extend(extract_leaf_nodes(subtree, new_path))
    
    return leaves

def normalize_question_text(text: str) -> str:
    """Normalize question text for comparison."""
    # Remove "Question: " prefix if present
    text = re.sub(r'^Question:\s*', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    # Remove trailing periods and question marks for comparison
    text = text.rstrip('.?')
    return text.lower().strip()

def create_question_to_index_mapping(subject_data: Dict[str, Dict[str, str]]) -> Dict[str, tuple]:
    """Create mapping from normalized question text to (subject, index)."""
    question_to_index = {}
    
    for subject, questions in subject_data.items():
        for index, question_text in questions.items():
            normalized = normalize_question_text(question_text)
            question_to_index[normalized] = (subject, index)
    
    return question_to_index

def find_question_index(input_text: str, question_mapping: Dict[str, tuple]) -> tuple:
    """Find the subject and index for a given question input."""
    normalized_input = normalize_question_text(input_text)
    
    # Direct match
    if normalized_input in question_mapping:
        return question_mapping[normalized_input]
    
    # Try to find partial matches (for truncated questions)
    for question_norm, (subject, index) in question_mapping.items():
        if len(normalized_input) > 50 and normalized_input in question_norm:
            return (subject, index)
        elif len(question_norm) > 50 and question_norm in normalized_input:
            return (subject, index)
    
    return None, None

def rebuild_tree_with_dove_scores(original_tree: Dict[Any, Any], 
                                  leaf_nodes: List[Dict[Any, Any]], 
                                  dove_data: Dict[str, float],
                                  question_mapping: Dict[str, tuple]) -> Dict[Any, Any]:
    """Rebuild the tree structure including only nodes that have DOVE scores."""
    
    # Create mapping of tree paths to leaf nodes with DOVE scores
    valid_paths = set()
    leaf_dove_mapping = {}
    
    total_leaves = len(leaf_nodes)
    matched_leaves = 0
    dove_matched_leaves = 0
    
    print(f"Processing {total_leaves} leaf nodes...")
    
    for leaf in leaf_nodes:
        # Find the question index
        subject, index = find_question_index(leaf['input'], question_mapping)
        
        if subject and index:
            matched_leaves += 1
            # Check if DOVE score exists for this index
            if index in dove_data:
                dove_matched_leaves += 1
                leaf_copy = leaf.copy()
                leaf_copy['dove_score'] = dove_data[index]
                leaf_copy['subject'] = subject
                leaf_copy['question_index'] = index
                
                path_key = tuple(leaf['tree_path'])
                leaf_dove_mapping[path_key] = leaf_copy
                valid_paths.add(path_key)
                
                # Add all parent paths to valid_paths
                for i in range(len(leaf['tree_path'])):
                    parent_path = tuple(leaf['tree_path'][:i])
                    valid_paths.add(parent_path)
    
    print(f"Matched {matched_leaves}/{total_leaves} leaves to question mapping")
    print(f"Found DOVE scores for {dove_matched_leaves}/{matched_leaves} matched leaves")
    print(f"Coverage: {dove_matched_leaves/total_leaves*100:.1f}% of original tree")
    
    def filter_tree(node: Dict[Any, Any], path: tuple = ()) -> Dict[Any, Any]:
        """Recursively filter tree to include only valid paths."""
        if path not in valid_paths:
            return None
        
        # If this is a leaf node with DOVE score
        if path in leaf_dove_mapping:
            return leaf_dove_mapping[path]
        
        # Otherwise, process internal node
        new_node = {k: v for k, v in node.items() if k != 'subtrees'}
        
        if 'subtrees' in node and isinstance(node['subtrees'], list):
            new_subtrees = []
            for i, subtree in enumerate(node['subtrees']):
                child_path = path + (str(i),)
                filtered_subtree = filter_tree(subtree, child_path)
                if filtered_subtree is not None:
                    new_subtrees.append(filtered_subtree)
            
            if new_subtrees:
                new_node['subtrees'] = new_subtrees
                # Update size to reflect filtered tree
                new_node['size'] = sum(
                    child.get('size', 1) if isinstance(child.get('subtrees'), (int, type(None)))
                    else child.get('size', 0)
                    for child in new_subtrees
                )
            else:
                return None
        
        return new_node
    
    return filter_tree(original_tree)

def main():
    print("Loading JSON files...")
    
    # Load the three JSON files
    try:
        with open('data/MMLU.json', 'r') as f:
            mmlu_data = json.load(f)
        
        with open('data/MMLU_DOVE.json', 'r') as f:
            dove_data = json.load(f)
        
        with open('data/mmlu_question_subject.json', 'r') as f:
            subject_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        sys.exit(1)
    
    print(f"Loaded MMLU tree with {mmlu_data.get('size', 0)} total questions")
    print(f"Loaded {len(dove_data)} DOVE scores")
    print(f"Loaded {len(subject_data)} subjects in question mapping")
    
    # Extract leaf nodes from MMLU tree
    print("\nExtracting leaf nodes from MMLU tree...")
    leaf_nodes = extract_leaf_nodes(mmlu_data)
    print(f"Found {len(leaf_nodes)} leaf nodes")
    
    # Create question-to-index mapping
    print("\nCreating question-to-index mapping...")
    question_mapping = create_question_to_index_mapping(subject_data)
    print(f"Created mapping for {len(question_mapping)} questions")
    
    # Rebuild tree with DOVE scores
    print("\nRebuilding tree with DOVE scores...")
    combined_tree = rebuild_tree_with_dove_scores(
        mmlu_data, leaf_nodes, dove_data, question_mapping
    )
    
    if combined_tree is None:
        print("Error: No valid tree structure could be created")
        sys.exit(1)
    
    # Add metadata about the filtering
    combined_tree['metadata'] = {
        'original_size': mmlu_data.get('size', 0),
        'filtered_size': combined_tree.get('size', 0),
        'dove_coverage': len(dove_data),
        'total_subjects': len(subject_data),
        'creation_info': 'Combined MMLU tree with DOVE scores, filtered to remove missing values'
    }
    
    # Save the combined tree
    output_file = 'data/MMLU_combined_with_DOVE.json'
    print(f"\nSaving combined tree to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(combined_tree, f, indent=2)
    
    print("Done!")
    print(f"Original tree size: {mmlu_data.get('size', 0)}")
    print(f"Combined tree size: {combined_tree.get('size', 0)}")
    print(f"Reduction: {(1 - combined_tree.get('size', 0)/mmlu_data.get('size', 1))*100:.1f}%")

if __name__ == "__main__":
    main() 