#!/usr/bin/env python3
"""
DOVE Perturbation Analysis and Examples

This script documents DOVE's perturbation methodology and provides inferred examples
based on common robustness testing practices for language models.
"""

def document_dove_methodology():
    """Document DOVE's perturbation approach based on available information"""
    
    methodology = {
        "perturbation_types": [
            {
                "type": "Semantic Preservation",
                "description": "DOVE creates 'semantically equivalent variations' that preserve meaning",
                "examples": [
                    {
                        "original": "What is the capital of France?",
                        "perturbations": [
                            "What city serves as the capital of France?",
                            "Which city is France's capital?",
                            "France's capital city is what?",
                            "What is the name of France's capital?"
                        ]
                    }
                ]
            },
            {
                "type": "Syntactic Variation", 
                "description": "Changes sentence structure while preserving content",
                "examples": [
                    {
                        "original": "If x = 2 and y = 3, what is x + y?",
                        "perturbations": [
                            "Given that x equals 2 and y equals 3, calculate x + y.",
                            "With x = 2 and y = 3 provided, find the sum x + y.",
                            "What is the result of x + y when x = 2 and y = 3?",
                            "Calculate the value of x + y, where x = 2 and y = 3."
                        ]
                    }
                ]
            },
            {
                "type": "Lexical Substitution",
                "description": "Replaces words with synonyms or alternative phrasings",
                "examples": [
                    {
                        "original": "Which element has the chemical symbol 'O'?",
                        "perturbations": [
                            "What element is represented by the chemical symbol 'O'?",
                            "Which chemical element uses the symbol 'O'?",
                            "The chemical symbol 'O' represents which element?",
                            "What element corresponds to the symbol 'O'?"
                        ]
                    }
                ]
            }
        ],
        
        "robustness_calculation": {
            "formula": "Robustness(q) = Σ(correct_responses) / total_perturbations",
            "explanation": "For each question, DOVE calculates the fraction of perturbations where the model gives correct answers",
            "score_range": "0.0 (fails all perturbations) to 1.0 (succeeds on all perturbations)"
        },
        
        "key_principles": [
            "Non-semantic perturbations: Changes don't alter the fundamental meaning",
            "Preserve correctness: The correct answer remains the same across perturbations", 
            "Test brittleness: Reveals when models rely on specific phrasing patterns",
            "Measure consistency: Robust models should perform similarly across variations"
        ]
    }
    
    return methodology

def analyze_mmlu_perturbation_examples():
    """Analyze potential DOVE perturbations for MMLU-style questions"""
    
    examples = [
        {
            "domain": "College Mathematics",
            "original_question": "Find the degree for the given field extension Q(√2, √3, √18) over Q.",
            "robustness_score": 0.290,  # From our data
            "inferred_perturbations": [
                "Determine the degree of the field extension Q(√2, √3, √18) over Q.",
                "What is the degree of Q(√2, √3, √18) as an extension of Q?",
                "Calculate the degree of the field extension Q(√2, √3, √18)/Q.",
                "For the field extension Q(√2, √3, √18) over Q, find its degree."
            ],
            "why_fragile": "Mathematical notation and terminology variations confuse the model"
        },
        
        {
            "domain": "High School Physics", 
            "original_question": "A ball is thrown vertically upward. What happens to its acceleration at the highest point?",
            "robustness_score": 0.301,  # From our data
            "inferred_perturbations": [
                "When a ball is thrown straight up, what is its acceleration at the peak?",
                "At the highest point of its trajectory, what acceleration does an upward-thrown ball experience?",
                "What acceleration occurs when a vertically thrown ball reaches maximum height?",
                "A ball thrown upward reaches its highest point. What is its acceleration there?"
            ],
            "why_fragile": "Physics concepts require understanding beyond surface-level pattern matching"
        },
        
        {
            "domain": "Abstract Algebra",
            "original_question": "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the index of <p> in S_5.",
            "robustness_score": 0.291,  # From our data  
            "inferred_perturbations": [
                "Given p = (1, 2, 5, 4)(2, 3) as an element of S_5, determine the index of <p> in S_5.",
                "For the permutation p = (1, 2, 5, 4)(2, 3) in S_5, calculate the index of the subgroup <p>.",
                "What is the index of the cyclic subgroup <p> in S_5, where p = (1, 2, 5, 4)(2, 3)?",
                "In S_5, if p = (1, 2, 5, 4)(2, 3), find the index of the subgroup generated by p."
            ],
            "why_fragile": "Complex algebraic notation and precise mathematical language requirements"
        }
    ]
    
    return examples

def generate_perturbation_report():
    """Generate comprehensive report on DOVE perturbation methodology"""
    
    print("=" * 80)
    print("DOVE PERTURBATION METHODOLOGY ANALYSIS")
    print("=" * 80)
    
    methodology = document_dove_methodology()
    
    print("\n1. PERTURBATION TYPES:")
    print("-" * 40)
    
    for ptype in methodology["perturbation_types"]:
        print(f"\n{ptype['type']}:")
        print(f"  Description: {ptype['description']}")
        
        for example in ptype["examples"]:
            print(f"\n  Original: {example['original']}")
            print("  Perturbations:")
            for i, pert in enumerate(example["perturbations"], 1):
                print(f"    {i}. {pert}")
    
    print(f"\n2. ROBUSTNESS CALCULATION:")
    print("-" * 40)
    print(f"Formula: {methodology['robustness_calculation']['formula']}")
    print(f"Explanation: {methodology['robustness_calculation']['explanation']}")
    print(f"Range: {methodology['robustness_calculation']['score_range']}")
    
    print(f"\n3. KEY PRINCIPLES:")
    print("-" * 40)
    for principle in methodology["key_principles"]:
        print(f"  • {principle}")
    
    print(f"\n" + "=" * 80)
    print("MMLU-SPECIFIC PERTURBATION ANALYSIS")
    print("=" * 80)
    
    examples = analyze_mmlu_perturbation_examples()
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['domain'].upper()}")
        print(f"   Robustness Score: {example['robustness_score']:.3f}")
        print("-" * 50)
        print(f"Original: {example['original_question']}")
        print("\nInferred DOVE Perturbations:")
        
        for j, pert in enumerate(example['inferred_perturbations'], 1):
            print(f"  {j}. {pert}")
        
        print(f"\nWhy This Domain is Fragile:")
        print(f"  {example['why_fragile']}")
    
    print(f"\n" + "=" * 80)
    print("IMPLICATIONS FOR HIERARCHICAL EVALUATION")
    print("=" * 80)
    
    print("\nWhy DOVE + EvalTree Integration Matters:")
    print("  • Reveals that mathematical domains show systematic brittleness")
    print("  • Question-level perturbations → Subject-level weakness patterns")
    print("  • Hierarchical aggregation uncovers domain-specific vulnerabilities")
    print("  • Traditional accuracy misses these nuanced robustness failures")
    
    print(f"\nOur Contribution:")
    print("  • First to aggregate DOVE scores hierarchically")
    print("  • Demonstrates mathematical domain clustering at low robustness")
    print("  • Shows different error patterns vs traditional accuracy evaluation")

if __name__ == "__main__":
    generate_perturbation_report() 