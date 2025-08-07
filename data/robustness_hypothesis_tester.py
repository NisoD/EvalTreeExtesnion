#!/usr/bin/env python3
"""
Robustness vs Accuracy Hypothesis Tester

This script tests the hypothesis that DOVE robustness is a better predictor
of model failures than accuracy alone by generating targeted questions.

HYPOTHESIS:
- Low Accuracy + Low Robustness → Model FAILS on new questions
- Low Accuracy + High Robustness → Model SUCCEEDS on new questions
"""

import json
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import openai
from together import Together


class RobustnessHypothesisTester:
    """Tests whether robustness predicts failures better than accuracy."""

    def __init__(self, openai_api_key: str = None, together_api_key: str = None):
        """Initialize with API keys."""
        self.openai_api_key = openai_api_key
        self.together_api_key = ()

        if self.together_api_key:
            self.together_client = Together(api_key=self.together_api_key)
        else:
            self.together_client = None

    def load_combined_tree(self, tree_path: str) -> Dict[str, Any]:
        """Load the combined MMLU tree with DOVE scores."""
        with open(tree_path, "r") as f:
            return json.load(f)

    def find_accuracy_robustness_patterns(
        self, tree: Dict[str, Any], model_name: str = "meta-llama/Llama-3-8b-chat-hf"
    ) -> Dict[str, List[Dict]]:
        """Find questions with different accuracy/robustness patterns."""

        patterns = {
            "low_acc_low_rob": [],  # Should FAIL on new questions
            "low_acc_high_rob": [],  # Should SUCCEED on new questions
            "high_acc_low_rob": [],  # Accurate but brittle
            "high_acc_high_rob": [],  # Strong performance
        }

        def extract_questions(node, capability_path=""):
            if isinstance(node.get("subtrees"), (int, type(None))):
                # Leaf node with question
                if "input" in node and "dove_score" in node and "ranking" in node:
                    # Find model's accuracy
                    accuracy = None
                    for model_data in node["ranking"]:
                        if model_data[0] == model_name:
                            accuracy = model_data[1]
                            break

                    if accuracy is not None:
                        dove_score = node["dove_score"]

                        question_data = {
                            "question": node["input"],
                            "accuracy": accuracy,
                            "dove_score": dove_score,
                            "subject": node.get("subject", "unknown"),
                            "capability": capability_path,
                            "gap": abs(accuracy - dove_score),
                        }

                        # Classify based on accuracy and robustness
                        if accuracy < 0.5 and dove_score < 0.5:
                            patterns["low_acc_low_rob"].append(question_data)
                        elif accuracy < 0.5 and dove_score >= 0.5:
                            patterns["low_acc_high_rob"].append(question_data)
                        elif accuracy >= 0.5 and dove_score < 0.5:
                            patterns["high_acc_low_rob"].append(question_data)
                        else:
                            patterns["high_acc_high_rob"].append(question_data)

            elif isinstance(node.get("subtrees"), list):
                # Internal node - recurse
                current_capability = node.get("capability", capability_path)
                for child in node["subtrees"]:
                    extract_questions(child, current_capability)

        extract_questions(tree)

        # If no patterns found, let's try a more aggressive search
        if all(len(patterns[k]) == 0 for k in patterns):
            print(
                "🔍 No patterns found with standard search, trying deeper traversal..."
            )

            def deep_search(node, depth=0):
                """More aggressive search through the tree."""
                if depth > 10:  # Prevent infinite recursion
                    return

                # Check all possible fields for question data
                if isinstance(node, dict):
                    if "input" in node and "dove_score" in node and "ranking" in node:
                        # Found a question!
                        accuracy = None
                        for model_data in node["ranking"]:
                            if model_data[0] == model_name:
                                accuracy = model_data[1]
                                break

                        if accuracy is not None:
                            dove_score = node["dove_score"]

                            question_data = {
                                "question": node["input"],
                                "accuracy": accuracy,
                                "dove_score": dove_score,
                                "subject": node.get("subject", "unknown"),
                                "capability": node.get("capability", "unknown"),
                                "gap": abs(accuracy - dove_score),
                            }

                            # Classify
                            if accuracy < 0.5 and dove_score < 0.5:
                                patterns["low_acc_low_rob"].append(question_data)
                            elif accuracy < 0.5 and dove_score >= 0.5:
                                patterns["low_acc_high_rob"].append(question_data)
                            elif accuracy >= 0.5 and dove_score < 0.5:
                                patterns["high_acc_low_rob"].append(question_data)
                            else:
                                patterns["high_acc_high_rob"].append(question_data)

                    # Continue searching in all dictionary values
                    for key, value in node.items():
                        if isinstance(value, (dict, list)):
                            deep_search(value, depth + 1)

                elif isinstance(node, list):
                    for item in node:
                        deep_search(item, depth + 1)

            deep_search(tree)
            print(
                f"📊 After deep search - found {sum(len(v) for v in patterns.values())} total questions"
            )

        return patterns

    def generate_concise_question_prompt(
        self, capability: str, example_questions: List[str], num_examples: int = 2
    ) -> Tuple[str, str]:
        """Generate prompt for concise questions with few-shot examples."""

        system_prompt = """You are a mathematics question generator. Create questions that require only a single, concise answer (like A, B, C, D or a number). Follow the examples exactly for format and length."""

        # Few-shot examples for concise answers
        few_shot_examples = """
Example 1:
Q: What is 2 + 3?
A: 5

Example 2: 
Q: Which is larger: 0.7 or 0.3?
A: 0.7

Example 3:
Q: True or False: All squares are rectangles.
A: True
"""

        user_prompt = f"""Generate ONE mathematics question for this capability: {capability}

{few_shot_examples}

Reference questions (understand the topic, don't copy):
{chr(10).join([f"- {ex[:100]}..." for ex in example_questions[:num_examples]])}

Requirements:
- Question should have ONE clear, short answer
- Use multiple choice (A/B/C/D) OR True/False OR single number
- Keep question under 100 words
- Generate ONLY the question, nothing else"""

        return system_prompt, user_prompt

    def generate_question_via_openai(
        self, capability: str, example_questions: List[str]
    ) -> str:
        """Generate a concise question using GPT-4o-mini."""

        system_prompt, user_prompt = self.generate_concise_question_prompt(
            capability, example_questions
        )

        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def evaluate_question_concisely(
        self, question: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo"
    ) -> Dict[str, Any]:
        """Evaluate question and get concise answer."""

        try:
            response = self.together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Answer this question concisely. Give only the final answer (like A, B, C, D, True/False, or a number).

Question: {question}

Answer:""",
                    }
                ],
                max_tokens=50,
                temperature=0.1,
            )

            answer = response.choices[0].message.content.strip()

            return {"question": question, "model_answer": answer, "success": True}

        except Exception as e:
            return {
                "question": question,
                "model_answer": None,
                "success": False,
                "error": str(e),
            }

    def test_robustness_hypothesis(
        self, tree_path: str, num_questions_per_pattern: int = 3
    ) -> Dict[str, Any]:
        """Test the main hypothesis: robustness predicts failures better than accuracy."""

        print("🔍 Loading combined tree and finding accuracy/robustness patterns...")
        tree = self.load_combined_tree(tree_path)
        patterns = self.find_accuracy_robustness_patterns(tree)

        print(f"📊 Pattern Statistics:")
        for pattern_name, questions in patterns.items():
            print(f"  {pattern_name}: {len(questions)} questions")

        # Test hypothesis on each pattern
        results = {
            "hypothesis": "Low Accuracy + Low Robustness → FAIL, Low Accuracy + High Robustness → SUCCEED",
            "patterns_tested": {},
            "overall_results": {},
        }

        for pattern_name, questions in patterns.items():
            if len(questions) == 0:
                continue

            print(f"\n🎯 Testing pattern: {pattern_name}")

            # Sample questions for this pattern
            sample_questions = random.sample(
                questions, min(num_questions_per_pattern, len(questions))
            )

            pattern_results = {
                "pattern_description": pattern_name,
                "sample_size": len(sample_questions),
                "original_stats": {
                    "avg_accuracy": sum(q["accuracy"] for q in sample_questions)
                    / len(sample_questions),
                    "avg_dove_score": sum(q["dove_score"] for q in sample_questions)
                    / len(sample_questions),
                    "avg_gap": sum(q["gap"] for q in sample_questions)
                    / len(sample_questions),
                },
                "generated_questions": [],
                "evaluations": [],
            }

            for i, q_data in enumerate(sample_questions):
                print(
                    f"  Generating question {i+1}/{len(sample_questions)} for {q_data['capability'][:50]}..."
                )

                # Generate new question based on this capability
                generated_q = self.generate_question_via_openai(
                    q_data["capability"], [q_data["question"]]
                )

                if generated_q:
                    pattern_results["generated_questions"].append(
                        {
                            "original_question": q_data["question"][:100] + "...",
                            "generated_question": generated_q,
                            "original_accuracy": q_data["accuracy"],
                            "original_dove_score": q_data["dove_score"],
                            "capability": q_data["capability"],
                        }
                    )

                    print(f"    ✓ Generated: {generated_q[:60]}...")

                    # Evaluate the generated question
                    print(f"    🔍 Evaluating...")
                    eval_result = self.evaluate_question_concisely(generated_q)
                    pattern_results["evaluations"].append(eval_result)

                    if eval_result["success"]:
                        print(f"    📝 Answer: {eval_result['model_answer']}")
                    else:
                        print(f"    ❌ Evaluation failed")
                else:
                    print(f"    ❌ Generation failed")

                time.sleep(1)  # Rate limiting

            results["patterns_tested"][pattern_name] = pattern_results

        return results

    def analyze_hypothesis_results(self, results: Dict[str, Any]) -> None:
        """Analyze results to validate the robustness hypothesis."""

        print("\n" + "=" * 80)
        print("🎯 ROBUSTNESS vs ACCURACY HYPOTHESIS ANALYSIS")
        print("=" * 80)

        print(f"\nHypothesis: {results['hypothesis']}")

        for pattern_name, pattern_data in results["patterns_tested"].items():
            print(f"\n📊 PATTERN: {pattern_name.upper().replace('_', ' ')}")
            print(f"   Sample size: {pattern_data['sample_size']}")
            print(
                f"   Original avg accuracy: {pattern_data['original_stats']['avg_accuracy']:.3f}"
            )
            print(
                f"   Original avg DOVE score: {pattern_data['original_stats']['avg_dove_score']:.3f}"
            )
            print(
                f"   Original avg gap: {pattern_data['original_stats']['avg_gap']:.3f}"
            )

            successful_evals = [e for e in pattern_data["evaluations"] if e["success"]]
            print(
                f"   Generated questions evaluated: {len(successful_evals)}/{pattern_data['sample_size']}"
            )

            if successful_evals:
                print(f"   Sample answers:")
                for i, eval_data in enumerate(successful_evals[:2]):
                    print(f"     Q{i+1}: {eval_data['model_answer']}")

        # Hypothesis validation summary
        print(f"\n🎯 HYPOTHESIS VALIDATION SUMMARY:")

        low_acc_low_rob = results["patterns_tested"].get("low_acc_low_rob", {})
        low_acc_high_rob = results["patterns_tested"].get("low_acc_high_rob", {})

        if low_acc_low_rob and low_acc_high_rob:
            print(f"✅ Both critical patterns tested:")
            print(
                f"   - Low Acc + Low Rob: {len(low_acc_low_rob.get('evaluations', []))} questions"
            )
            print(
                f"   - Low Acc + High Rob: {len(low_acc_high_rob.get('evaluations', []))} questions"
            )
            print(
                f"\n🔍 Next step: Manual evaluation needed to determine success/failure rates"
            )
        else:
            print(f"⚠️  Missing critical patterns for hypothesis testing")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test robustness vs accuracy hypothesis"
    )
    parser.add_argument("--tree", required=True, help="Path to combined MMLU tree")
    parser.add_argument("--output", required=True, help="Output file prefix")
    parser.add_argument(
        "--num-questions", type=int, default=3, help="Questions per pattern"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = RobustnessHypothesisTester()

    # Run hypothesis test
    print("🚀 Starting robustness vs accuracy hypothesis test...")
    results = tester.test_robustness_hypothesis(args.tree, args.num_questions)

    # Save results
    output_file = f"{args.output}_robustness_hypothesis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {output_file}")

    # Analyze results
    tester.analyze_hypothesis_results(results)

    print(f"\n✅ Hypothesis testing complete!")


if __name__ == "__main__":
    main()
