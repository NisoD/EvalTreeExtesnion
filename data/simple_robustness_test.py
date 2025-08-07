#!/usr/bin/env python3
"""
Simple Robustness vs Accuracy Test

Uses our existing weakness profiles to test the hypothesis that 
robustness predicts failures better than accuracy alone.
"""

import json
import numpy as np
from collections import defaultdict

def load_weakness_profile(profile_path: str):
    """Load weakness profile."""
    with open(profile_path, 'r') as f:
        return json.load(f)

def analyze_robustness_patterns(profile_path: str):
    """Analyze accuracy vs robustness patterns in weakness areas."""
    
    print("🔍 Loading weakness profile...")
    profile = load_weakness_profile(profile_path)
    
    print(f"📊 Analyzing {len(profile['weakness_nodes'])} weakness areas...")
    
    # Extract all questions from weakness nodes
    all_questions = []
    
    for weakness in profile['weakness_nodes']:
        capability = weakness['capability']
        
        # Extract questions from this weakness node
        def extract_from_node(node):
            if isinstance(node.get('subtrees'), (int, type(None))):
                # Leaf node with question
                if 'input' in node and 'dove_score' in node and 'ranking' in node:
                    # Find Llama accuracy
                    accuracy = None
                    for model_data in node['ranking']:
                        if model_data[0] == "Llama-3.1-8B-Instruct":
                            accuracy = model_data[1]
                            break
                    
                    if accuracy is not None:
                        all_questions.append({
                            'capability': capability,
                            'question': node['input'],
                            'accuracy': accuracy,
                            'dove_score': node['dove_score'],
                            'subject': node.get('subject', 'unknown'),
                            'gap': abs(accuracy - node['dove_score'])
                        })
            
            elif isinstance(node.get('subtrees'), list):
                for child in node['subtrees']:
                    extract_from_node(child)
        
        extract_from_node(weakness['node_data'])
    
    print(f"✅ Extracted {len(all_questions)} questions from weakness areas")
    
    # Classify questions by accuracy/robustness patterns
    patterns = {
        "low_acc_low_rob": [],      # Should predict FAILURE
        "low_acc_high_rob": [],     # Should predict SUCCESS  
        "high_acc_low_rob": [],     # Accurate but brittle
        "high_acc_high_rob": []     # Strong performance
    }
    
    for q in all_questions:
        acc = q['accuracy']
        rob = q['dove_score']
        
        if acc < 0.5 and rob < 0.5:
            patterns["low_acc_low_rob"].append(q)
        elif acc < 0.5 and rob >= 0.5:
            patterns["low_acc_high_rob"].append(q)
        elif acc >= 0.5 and rob < 0.5:
            patterns["high_acc_low_rob"].append(q)
        else:
            patterns["high_acc_high_rob"].append(q)
    
    # Print results
    print("\n🎯 ROBUSTNESS vs ACCURACY PATTERNS:")
    print("="*60)
    
    for pattern_name, questions in patterns.items():
        if len(questions) > 0:
            avg_acc = np.mean([q['accuracy'] for q in questions])
            avg_rob = np.mean([q['dove_score'] for q in questions])
            avg_gap = np.mean([q['gap'] for q in questions])
            
            print(f"\n📊 {pattern_name.upper().replace('_', ' ')}:")
            print(f"   Count: {len(questions)}")
            print(f"   Avg Accuracy: {avg_acc:.3f}")
            print(f"   Avg DOVE Score: {avg_rob:.3f}")
            print(f"   Avg Gap: {avg_gap:.3f}")
            
            # Show sample capabilities
            capabilities = list(set([q['capability'][:50] + "..." for q in questions[:3]]))
            print(f"   Sample capabilities:")
            for cap in capabilities:
                print(f"     - {cap}")
    
    # Key insight: Which patterns exist?
    print(f"\n🔍 HYPOTHESIS TEST READINESS:")
    
    critical_patterns = ["low_acc_low_rob", "low_acc_high_rob"]
    ready_for_test = all(len(patterns[p]) > 0 for p in critical_patterns)
    
    if ready_for_test:
        print(f"✅ READY! We have both critical patterns:")
        print(f"   - Low Acc + Low Rob: {len(patterns['low_acc_low_rob'])} questions (predict FAILURE)")
        print(f"   - Low Acc + High Rob: {len(patterns['low_acc_high_rob'])} questions (predict SUCCESS)")
        print(f"\n🎯 Next step: Generate questions from these capabilities and test model performance!")
    else:
        print(f"⚠️  Missing critical patterns. Available:")
        for p in critical_patterns:
            print(f"   - {p}: {len(patterns[p])} questions")
    
    return patterns

def find_best_examples(patterns):
    """Find the best examples for each pattern."""
    
    print(f"\n🎯 BEST EXAMPLES FOR HYPOTHESIS TESTING:")
    print("="*60)
    
    for pattern_name, questions in patterns.items():
        if len(questions) > 0:
            print(f"\n📋 {pattern_name.upper().replace('_', ' ')}:")
            
            # Sort by gap (most extreme examples)
            sorted_questions = sorted(questions, key=lambda x: x['gap'], reverse=True)
            
            for i, q in enumerate(sorted_questions[:3]):  # Top 3
                print(f"   {i+1}. Capability: {q['capability'][:60]}...")
                print(f"      Accuracy: {q['accuracy']:.3f}, DOVE: {q['dove_score']:.3f}, Gap: {q['gap']:.3f}")
                print(f"      Subject: {q['subject']}")
                print(f"      Question: {q['question'][:80]}...")
                print()

def main():
    """Main analysis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_robustness_test.py <weakness_profile.json>")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    
    print("🚀 SIMPLE ROBUSTNESS vs ACCURACY HYPOTHESIS TEST")
    print("="*80)
    
    # Analyze patterns
    patterns = analyze_robustness_patterns(profile_path)
    
    # Find best examples
    find_best_examples(patterns)
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Use the capabilities above to generate targeted questions for hypothesis testing.")

if __name__ == "__main__":
    main() 