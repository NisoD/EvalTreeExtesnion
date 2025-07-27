#!/usr/bin/env python3
"""
Improved Weakness Profile Analyzer for MMLU DOVE Evaluation

This version properly handles missing DOVE scores to prevent false weakness detection.
Missing questions are excluded from analysis rather than treated as score 0.
"""

import json
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional

class ImprovedWeaknessAnalyzer:
    def __init__(self):
        self.dove_scores = {}
        self.eval_tree = {}
        self.missing_questions = set()
        
    def load_dove_scores(self, file_path: str) -> Dict[str, float]:
        """Load DOVE scores from JSON file"""
        print(f"Loading DOVE scores from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # Convert keys to strings, ensure values are floats
                scores = {}
                for k, v in data.items():
                    if v is not None and not (isinstance(v, str) and v.strip() == ""):
                        try:
                            scores[str(k)] = float(v)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid score for question {k}: {v}")
                            continue
                
                print(f"Loaded {len(scores)} DOVE scores")
                print(f"Score range: {min(scores.values()):.4f} to {max(scores.values()):.4f}")
                
                # Find the range of indices
                indices = [int(k) for k in scores.keys()]
                print(f"Index range: {min(indices)} to {max(indices)}")
                
                return scores
            else:
                print(f"Unexpected data format in {file_path}")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {}
        except Exception as e:
            print(f"Error loading DOVE scores: {e}")
            return {}
    
    def load_eval_tree_flexible(self, file_path: str) -> Dict:
        """Load EvalTree with flexible parsing to handle corrupted files"""
        print(f"Loading EvalTree from: {file_path}")
        
        try:
            # Try loading the full file first
            with open(file_path, 'r') as f:
                content = f.read()
            
            # If it's too large or corrupted, try to find a valid JSON portion
            if len(content) > 2000000:  # > 2MB, likely corrupted
                print("Large file detected, attempting to extract valid JSON...")
                # Find the first complete JSON object
                brace_count = 0
                valid_end = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            valid_end = i + 1
                            break
                
                if valid_end > 0:
                    content = content[:valid_end]
                    print(f"Extracted valid JSON portion ({valid_end} characters)")
            
            tree = json.loads(content)
            print(f"Successfully loaded EvalTree structure")
            
            # Print tree structure info
            if isinstance(tree, dict):
                print(f"Tree keys: {list(tree.keys())}")
                if 'size' in tree:
                    print(f"Tree size: {tree['size']}")
                if 'capability' in tree:
                    print(f"Root capability: {tree['capability'][:100]}...")
            
            return tree
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("Attempting to use the existing MMLU EvalTree from Datasets/...")
            
            # Fallback to the processed EvalTree if available
            fallback_path = "Datasets/MMLU/EvalTree/stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[located-split=[exclusion]10042-4000]/[dataset=MMLU]_[stage3-RecursiveClustering]_[split=10042-4000]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10].json"
            
            if os.path.exists(fallback_path):
                print(f"Using fallback tree: {fallback_path}")
                with open(fallback_path, 'r') as f:
                    return json.load(f)
            
            return {}
        except Exception as e:
            print(f"Error loading EvalTree: {e}")
            return {}
    
    def collect_leaf_indices(self, node, leaf_indices=None):
        """Recursively collect all leaf indices from the EvalTree"""
        if leaf_indices is None:
            leaf_indices = []
        
        if isinstance(node, dict):
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for subtree in subtrees:
                        self.collect_leaf_indices(subtree, leaf_indices)
                elif isinstance(subtrees, dict):
                    for subtree in subtrees.values():
                        self.collect_leaf_indices(subtree, leaf_indices)
                elif isinstance(subtrees, int):
                    leaf_indices.append(subtrees)
            elif 'description' in node and 'subtrees' in node:
                # Stage 4 format
                subtrees = node['subtrees']
                if isinstance(subtrees, int):
                    leaf_indices.append(subtrees)
                elif isinstance(subtrees, list):
                    for subtree in subtrees:
                        self.collect_leaf_indices(subtree, leaf_indices)
        elif isinstance(node, int):
            leaf_indices.append(node)
        elif isinstance(node, list):
            for item in node:
                self.collect_leaf_indices(item, leaf_indices)
        
        return leaf_indices
    
    def calculate_node_stats(self, node) -> Dict:
        """Calculate statistics for a node, properly handling missing DOVE scores"""
        leaf_indices = self.collect_leaf_indices(node)
        
        # CRITICAL: Only include indices that have DOVE scores
        # This prevents false weakness from missing questions
        scored_indices = [idx for idx in leaf_indices if str(idx) in self.dove_scores]
        missing_indices = [idx for idx in leaf_indices if str(idx) not in self.dove_scores]
        
        # Track missing questions for reporting
        self.missing_questions.update(missing_indices)
        
        if not scored_indices:
            return {
                'mean_score': None,
                'std_score': None,
                'min_score': None,
                'max_score': None,
                'count': 0,
                'total_questions': len(leaf_indices),
                'missing_count': len(missing_indices),
                'coverage': 0.0,
                'weakness_level': None,
                'reliable': False  # Not enough data for reliable analysis
            }
        
        scores = [self.dove_scores[str(idx)] for idx in scored_indices]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Coverage: percentage of questions in this capability that we have scores for
        coverage = len(scored_indices) / len(leaf_indices) if leaf_indices else 0
        
        # Determine if analysis is reliable (need good coverage and minimum questions)
        reliable = coverage >= 0.3 and len(scored_indices) >= 3
        
        # Determine weakness level based on mean score
        if mean_score < 0.3:
            weakness_level = "Critical"
        elif mean_score < 0.5:
            weakness_level = "High"
        elif mean_score < 0.7:
            weakness_level = "Moderate"
        else:
            weakness_level = "Low"
        
        return {
            'mean_score': round(mean_score, 4),
            'std_score': round(std_score, 4),
            'min_score': round(min_score, 4),
            'max_score': round(max_score, 4),
            'count': len(scored_indices),
            'total_questions': len(leaf_indices),
            'missing_count': len(missing_indices),
            'coverage': round(coverage, 4),
            'weakness_level': weakness_level,
            'reliable': reliable
        }
    
    def annotate_tree(self, node):
        """Recursively annotate tree with weakness statistics"""
        if isinstance(node, dict):
            # Calculate stats for this node
            stats = self.calculate_node_stats(node)
            node['weakness_stats'] = stats
            
            # Process subtrees
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for subtree in subtrees:
                        self.annotate_tree(subtree)
                elif isinstance(subtrees, dict):
                    for subtree in subtrees.values():
                        self.annotate_tree(subtree)
        
        return node
    
    def extract_capabilities(self, node, capabilities=None, path=""):
        """Extract all capabilities with their statistics"""
        if capabilities is None:
            capabilities = []
        
        if isinstance(node, dict) and 'weakness_stats' in node:
            stats = node['weakness_stats']
            # Only include reliable analyses to prevent false weaknesses
            if stats['reliable'] and stats['count'] > 0:
                capability_info = {
                    'path': path,
                    'capability': node.get('capability', node.get('description', 'Unknown capability')),
                    'mean_score': stats['mean_score'],
                    'weakness_level': stats['weakness_level'],
                    'count': stats['count'],
                    'total_questions': stats['total_questions'],
                    'missing_count': stats['missing_count'],
                    'coverage': stats['coverage'],
                    'reliable': stats['reliable'],
                    'std_score': stats['std_score'],
                    'min_score': stats['min_score'],
                    'max_score': stats['max_score']
                }
                capabilities.append(capability_info)
            
            # Process subtrees
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for i, subtree in enumerate(subtrees):
                        self.extract_capabilities(subtree, capabilities, f"{path}/{i}")
                elif isinstance(subtrees, dict):
                    for key, subtree in subtrees.items():
                        self.extract_capabilities(subtree, capabilities, f"{path}/{key}")
        
        return capabilities
    
    def generate_report(self, min_coverage=0.3, min_questions=3):
        """Generate comprehensive weakness report with proper missing data handling"""
        print("\nGenerating weakness profile...")
        print(f"Filtering criteria: min_coverage={min_coverage}, min_questions={min_questions}")
        
        # Reset missing questions tracker
        self.missing_questions = set()
        
        # Annotate tree with statistics
        annotated_tree = self.annotate_tree(self.eval_tree.copy())
        
        # Extract all capabilities
        all_capabilities = self.extract_capabilities(annotated_tree)
        
        # Filter capabilities with sufficient data and coverage
        valid_capabilities = [
            cap for cap in all_capabilities 
            if cap['reliable'] and cap['coverage'] >= min_coverage and cap['count'] >= min_questions
        ]
        
        # Sort by mean score (ascending = worst first)
        valid_capabilities.sort(key=lambda x: x['mean_score'])
        
        # Calculate overall statistics
        all_scores = list(self.dove_scores.values())
        overall_stats = {
            'total_questions_scored': len(self.dove_scores),
            'total_missing_questions': len(self.missing_questions),
            'total_capabilities_analyzed': len(valid_capabilities),
            'overall_mean_score': np.mean(all_scores) if all_scores else 0,
            'overall_std_score': np.std(all_scores) if all_scores else 0,
            'overall_min_score': np.min(all_scores) if all_scores else 0,
            'overall_max_score': np.max(all_scores) if all_scores else 0,
            'filtering_criteria': {
                'min_coverage': min_coverage,
                'min_questions': min_questions
            }
        }
        
        # Count by weakness level
        weakness_counts = {
            'Critical': len([c for c in valid_capabilities if c['weakness_level'] == 'Critical']),
            'High': len([c for c in valid_capabilities if c['weakness_level'] == 'High']),
            'Moderate': len([c for c in valid_capabilities if c['weakness_level'] == 'Moderate']),
            'Low': len([c for c in valid_capabilities if c['weakness_level'] == 'Low'])
        }
        
        report = {
            'summary': {**overall_stats, **weakness_counts},
            'top_weaknesses': valid_capabilities[:20],
            'weakness_by_level': {
                'Critical': [c for c in valid_capabilities if c['weakness_level'] == 'Critical'],
                'High': [c for c in valid_capabilities if c['weakness_level'] == 'High'],
                'Moderate': [c for c in valid_capabilities if c['weakness_level'] == 'Moderate'],
                'Low': [c for c in valid_capabilities if c['weakness_level'] == 'Low']
            },
            'all_capabilities': valid_capabilities,
            'data_quality': {
                'total_dove_scores': len(self.dove_scores),
                'total_missing_questions': len(self.missing_questions),
                'coverage_distribution': self._get_coverage_distribution(all_capabilities)
            }
        }
        
        return report
    
    def _get_coverage_distribution(self, capabilities):
        """Get distribution of coverage percentages"""
        if not capabilities:
            return {}
        
        coverages = [c['coverage'] for c in capabilities if c['coverage'] is not None]
        if not coverages:
            return {}
        
        return {
            'mean_coverage': round(np.mean(coverages), 4),
            'min_coverage': round(np.min(coverages), 4),
            'max_coverage': round(np.max(coverages), 4),
            'std_coverage': round(np.std(coverages), 4)
        }
    
    def print_summary(self, report):
        """Print human-readable summary with data quality information"""
        summary = report['summary']
        data_quality = report['data_quality']
        
        print("\n" + "="*80)
        print("MMLU WEAKNESS PROFILE SUMMARY (IMPROVED)")
        print("="*80)
        
        print(f"📊 DATA OVERVIEW:")
        print(f"  Questions with DOVE scores: {summary['total_questions_scored']:,}")
        print(f"  Questions missing scores:   {summary['total_missing_questions']:,}")
        print(f"  Capability areas analyzed:  {summary['total_capabilities_analyzed']}")
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"  Mean Score: {summary['overall_mean_score']:.4f} ± {summary['overall_std_score']:.4f}")
        print(f"  Score Range: {summary['overall_min_score']:.4f} - {summary['overall_max_score']:.4f}")
        
        print(f"\n🎯 FILTERING CRITERIA:")
        criteria = summary['filtering_criteria']
        print(f"  Minimum Coverage: {criteria['min_coverage']*100:.0f}%")
        print(f"  Minimum Questions: {criteria['min_questions']}")
        
        print(f"\n⚠️  WEAKNESS DISTRIBUTION:")
        print(f"  Critical (< 30%):   {summary['Critical']:3d} areas")
        print(f"  High (30-49%):      {summary['High']:3d} areas")
        print(f"  Moderate (50-69%):  {summary['Moderate']:3d} areas")
        print(f"  Low (≥ 70%):        {summary['Low']:3d} areas")
        
        print(f"\n🔍 TOP 10 WEAKEST CAPABILITY AREAS:")
        print("-" * 80)
        
        for i, weakness in enumerate(report['top_weaknesses'][:10], 1):
            coverage_pct = weakness['coverage'] * 100
            print(f"{i:2d}. Score: {weakness['mean_score']:.3f} ± {weakness['std_score']:.3f} | "
                  f"Level: {weakness['weakness_level']:8s} | "
                  f"Q: {weakness['count']:3d}/{weakness['total_questions']:3d} "
                  f"({coverage_pct:.0f}%)")
            print(f"    {weakness['capability'][:70]}...")
            print()
        
        # Data quality warning
        if summary['total_missing_questions'] > summary['total_questions_scored']:
            print("⚠️  WARNING: More questions are missing DOVE scores than have scores.")
            print("   This may affect the reliability of the weakness analysis.")
    
    def analyze(self, dove_file="MMLU_DOVE.json", tree_file="MMLU.json", 
                min_coverage=0.3, min_questions=3):
        """Main analysis function with improved missing data handling"""
        try:
            # Load data
            self.dove_scores = self.load_dove_scores(dove_file)
            self.eval_tree = self.load_eval_tree_flexible(tree_file)
            
            if not self.dove_scores:
                print("❌ No DOVE scores loaded. Please check your data file.")
                return None
            
            if not self.eval_tree:
                print("❌ No EvalTree loaded. Please check your tree file.")
                return None
            
            # Generate report
            report = self.generate_report(min_coverage, min_questions)
            
            # Print summary
            self.print_summary(report)
            
            # Save reports
            with open('weakness_profile_improved.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            summary_report = {
                'summary': report['summary'],
                'top_20_weaknesses': report['top_weaknesses'],
                'critical_areas': report['weakness_by_level']['Critical'],
                'data_quality': report['data_quality']
            }
            
            with open('weakness_profile_summary_improved.json', 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            print(f"\n✅ Analysis completed!")
            print(f"📄 Detailed report: weakness_profile_improved.json")
            print(f"📄 Summary report: weakness_profile_summary_improved.json")
            
            return report
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution"""
    analyzer = ImprovedWeaknessAnalyzer()
    
    print("🔍 MMLU Improved Weakness Profile Analyzer")
    print("="*50)
    print("✅ Properly handles missing DOVE scores")
    print("✅ Prevents false weakness detection")
    print("✅ Flexible tree format support")
    
    # Run analysis with conservative filtering to ensure reliability
    report = analyzer.analyze(
        dove_file="MMLU_DOVE.json",
        tree_file="MMLU.json",
        min_coverage=0.3,  # Require 30% coverage
        min_questions=3    # Require at least 3 questions
    )
    
    if report:
        print("\n🎉 Analysis completed successfully!")
        print("📊 Check the generated JSON files for detailed results.")
    else:
        print("\n❌ Analysis failed. Please check your data files.")

if __name__ == "__main__":
    main()