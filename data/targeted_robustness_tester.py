#!/usr/bin/env python3
"""
Targeted Robustness Hypothesis Tester

Uses the patterns we found in the weakness profile to generate and test questions
that validate whether robustness predicts model performance better than accuracy.
"""

import json
import random
import time
from typing import Any, Dict, List

import openai
from together import Together


class TargetedRobustnessTester:
    """Tests robustness hypothesis using weakness profile patterns."""

    def __init__(self, openai_api_key: str = None, together_api_key: str = None):
        """Initialize with API keys."""
        self.openai_api_key = openai_api_key
        self.together_api_key = ()

        if self.together_api_key:
            self.together_client = Together(api_key=self.together_api_key)

    def extract_patterns_from_weakness_profile(self, profile_path: str):
        """Extract the robustness patterns we found."""

        with open(profile_path, "r") as f:
            profile = json.load(f)

        patterns = {
            "low_rob_capabilities": [],  # Should predict FAILURE
            "high_rob_capabilities": [],  # Should predict SUCCESS
        }

        all_questions = []

        for weakness in profile["weakness_nodes"]:
            capability = weakness["capability"]

            # Extract questions from this weakness node
            def extract_from_node(node):
                if isinstance(node.get("subtrees"), (int, type(None))):
                    if "input" in node and "dove_score" in node and "ranking" in node:
                        # Find Llama accuracy
                        accuracy = None
                        for model_data in node["ranking"]:
                            if model_data[0] == "Llama-3.1-8B-Instruct":
                                accuracy = model_data[1]
                                break

                        if accuracy is not None:
                            all_questions.append(
                                {
                                    "capability": capability,
                                    "question": node["input"],
                                    "accuracy": accuracy,
                                    "dove_score": node["dove_score"],
                                    "subject": node.get("subject", "unknown"),
                                }
                            )

                elif isinstance(node.get("subtrees"), list):
                    for child in node["subtrees"]:
                        extract_from_node(child)

            extract_from_node(weakness["node_data"])

        # Find individual questions with different robustness patterns
        low_rob_questions = []
        high_rob_questions = []

        for q in all_questions:
            # Focus on low accuracy questions only (as per hypothesis)
            if q["accuracy"] <= 0.5:
                if q["dove_score"] < 0.5:
                    low_rob_questions.append(q)
                else:
                    high_rob_questions.append(q)

        print(
            f"  Debug: Found {len(low_rob_questions)} low-rob questions, {len(high_rob_questions)} high-rob questions"
        )

        # Group by capability for testing
        if low_rob_questions:
            # Group low robustness questions by capability
            low_rob_by_cap = {}
            for q in low_rob_questions:
                cap = q["capability"]
                if cap not in low_rob_by_cap:
                    low_rob_by_cap[cap] = []
                low_rob_by_cap[cap].append(q)

            # Take the capability with most low-rob questions
            best_low_cap = max(
                low_rob_by_cap.keys(), key=lambda c: len(low_rob_by_cap[c])
            )
            low_questions = low_rob_by_cap[best_low_cap]

            patterns["low_rob_capabilities"].append(
                {
                    "capability": best_low_cap,
                    "avg_dove": sum(q["dove_score"] for q in low_questions)
                    / len(low_questions),
                    "avg_accuracy": sum(q["accuracy"] for q in low_questions)
                    / len(low_questions),
                    "sample_questions": low_questions[:3],
                }
            )

        if high_rob_questions:
            # Group high robustness questions by capability
            high_rob_by_cap = {}
            for q in high_rob_questions:
                cap = q["capability"]
                if cap not in high_rob_by_cap:
                    high_rob_by_cap[cap] = []
                high_rob_by_cap[cap].append(q)

            # Take the capability with most high-rob questions
            best_high_cap = max(
                high_rob_by_cap.keys(), key=lambda c: len(high_rob_by_cap[c])
            )
            high_questions = high_rob_by_cap[best_high_cap]

            patterns["high_rob_capabilities"].append(
                {
                    "capability": best_high_cap,
                    "avg_dove": sum(q["dove_score"] for q in high_questions)
                    / len(high_questions),
                    "avg_accuracy": sum(q["accuracy"] for q in high_questions)
                    / len(high_questions),
                    "sample_questions": high_questions[:3],
                }
            )

        return patterns

    def generate_concise_question(
        self, capability: str, example_questions: List[str]
    ) -> str:
        """Generate a challenging question with few-shot prompting."""

        system_prompt = """You are an expert mathematics question generator. Create challenging, university-level questions that require deep understanding. Use multiple choice format (A/B/C/D) with one correct answer."""

        # Hard few-shot examples matching MMLU difficulty
        few_shot_examples = """
Example 1:
Q: Statement 1 | Every finite group of order p^2 where p is prime is abelian. Statement 2 | Every group of order 15 is cyclic.
A. True, True
B. False, False  
C. True, False
D. False, True

Example 2:
Q: Let f(x) = x^3 - 6x^2 + 11x - 6. Which of the following is NOT a root of f(x)?
A. 1
B. 2
C. 3
D. 4

Example 3:
Q: In propositional logic, which of the following is equivalent to ¬(P ∧ Q)?
A. ¬P ∧ ¬Q
B. ¬P ∨ ¬Q
C. P ∨ Q
D. P ∧ ¬Q
"""

        user_prompt = f"""Generate ONE challenging multiple-choice question for this capability: {capability}

{few_shot_examples}

Reference questions (understand the difficulty level and topic, don't copy):
{chr(10).join([f"- {ex[:100]}..." for ex in example_questions[:2]])}

Requirements:
- University/graduate level difficulty
- Multiple choice with 4 options (A/B/C/D)
- Requires deep mathematical reasoning
- Similar complexity to the reference questions
- Generate ONLY the question with choices, no answer"""

        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def evaluate_question(self, question: str) -> Dict[str, Any]:
        """Evaluate question with Together AI using correct model."""

        try:
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",  # Correct model name
                messages=[
                    {
                        "role": "user",
                        "content": f"Answer this multiple choice question. Give only the letter (A, B, C, or D):\n\n{question}",
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )

            return {
                "question": question,
                "answer": response.choices[0].message.content.strip(),
                "success": True,
            }

        except Exception as e:
            return {
                "question": question,
                "answer": None,
                "success": False,
                "error": str(e),
            }

    def test_robustness_hypothesis(
        self, profile_path: str, questions_per_pattern: int = 3
    ):
        """Test the robustness hypothesis."""

        print("🔍 Extracting patterns from weakness profile...")
        patterns = self.extract_patterns_from_weakness_profile(profile_path)

        print(f"📊 Found patterns:")
        print(f"  Low robustness capabilities: {len(patterns['low_rob_capabilities'])}")
        print(
            f"  High robustness capabilities: {len(patterns['high_rob_capabilities'])}"
        )

        results = {
            "hypothesis": "Low robustness → FAIL, High robustness → SUCCEED",
            "test_results": {},
        }

        for pattern_name, capabilities in patterns.items():
            if not capabilities:
                continue

            print(f"\n🎯 Testing {pattern_name}...")
            pattern_results = []

            for cap_data in capabilities[:2]:  # Test top 2 capabilities per pattern
                capability = cap_data["capability"]
                print(f"  📝 Capability: {capability[:50]}...")
                print(
                    f"     Avg DOVE: {cap_data['avg_dove']:.3f}, Avg Acc: {cap_data['avg_accuracy']:.3f}"
                )

                # Generate questions for this capability
                example_questions = [
                    q["question"] for q in cap_data["sample_questions"]
                ]

                cap_results = {
                    "capability": capability,
                    "avg_dove": cap_data["avg_dove"],
                    "avg_accuracy": cap_data["avg_accuracy"],
                    "generated_questions": [],
                }

                for i in range(questions_per_pattern):
                    print(f"    Generating question {i+1}...")

                    generated_q = self.generate_concise_question(
                        capability, example_questions
                    )

                    if generated_q:
                        print(f"      ✓ Generated: {generated_q[:50]}...")

                        # Evaluate the question
                        eval_result = self.evaluate_question(generated_q)

                        if eval_result["success"]:
                            print(f"      📝 Model answer: {eval_result['answer']}")
                        else:
                            print(f"      ❌ Evaluation failed")

                        cap_results["generated_questions"].append(
                            {"question": generated_q, "evaluation": eval_result}
                        )
                    else:
                        print(f"      ❌ Generation failed")

                    time.sleep(1)  # Rate limiting

                pattern_results.append(cap_results)

            results["test_results"][pattern_name] = pattern_results

        return results

    def analyze_results(self, results: Dict[str, Any]):
        """Analyze the hypothesis test results."""

        print("\n" + "=" * 80)
        print("🎯 ROBUSTNESS HYPOTHESIS TEST RESULTS")
        print("=" * 80)

        print(f"Hypothesis: {results['hypothesis']}")

        for pattern_name, pattern_data in results["test_results"].items():
            prediction = "FAILURE" if "low_rob" in pattern_name else "SUCCESS"

            print(
                f"\n📊 {pattern_name.upper().replace('_', ' ')} (Predict {prediction}):"
            )

            total_questions = 0
            successful_evaluations = 0

            for cap_data in pattern_data:
                print(f"   Capability: {cap_data['capability'][:50]}...")
                print(f"   Original DOVE: {cap_data['avg_dove']:.3f}")

                questions = cap_data["generated_questions"]
                successful = [q for q in questions if q["evaluation"]["success"]]

                total_questions += len(questions)
                successful_evaluations += len(successful)

                print(f"   Questions generated: {len(questions)}")
                print(f"   Successfully evaluated: {len(successful)}")

                if successful:
                    print(f"   Sample answers:")
                    for q in successful[:2]:
                        print(f"     - {q['evaluation']['answer']}")
                print()

        print(f"🎯 SUMMARY:")
        print(
            f"   Total questions generated: {sum(len(pd[0]['generated_questions']) + len(pd[1]['generated_questions']) if len(pd) > 1 else len(pd[0]['generated_questions']) for pd in results['test_results'].values() if pd)}"
        )
        print(
            f"   Hypothesis test: Ready for manual evaluation of success/failure rates!"
        )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test robustness hypothesis using weakness profiles"
    )
    parser.add_argument("--profile", required=True, help="Weakness profile path")
    parser.add_argument("--output", required=True, help="Output file prefix")
    parser.add_argument(
        "--questions", type=int, default=2, help="Questions per pattern"
    )

    args = parser.parse_args()

    print("🚀 TARGETED ROBUSTNESS HYPOTHESIS TEST")
    print("=" * 80)

    tester = TargetedRobustnessTester()
    results = tester.test_robustness_hypothesis(args.profile, args.questions)

    # Save results
    with open(f"{args.output}_targeted_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Analyze results
    tester.analyze_results(results)

    print(f"\n✅ Test complete! Results saved to {args.output}_targeted_results.json")


if __name__ == "__main__":
    main()
