# MMLU Weakness Profile Analysis Guide

## Overview
This tool maps your DOVE permutation-based evaluation scores to the existing MMLU EvalTree capability structure to identify specific areas of weakness.

## Required Files

### 1. DOVE Scores File (`MMLU_DOVE.json`)
**Format:** JSON object with question indices as keys and scores as values
```json
{
    "0": 0.208,
    "1": 0.2075,
    "2": 0.3635,
    "3": 0.1495,
    "4": 0.514,
    ...
}
```

### 2. EvalTree Structure File (`MMLU.json`)
**Format:** Hierarchical tree structure with capabilities
```json
{
    "capability": "Root capability description",
    "size": 14042,
    "depth": 1,
    "subtrees": [...]
}
```

## Usage

### Basic Usage
```bash
python flexible_weakness_analyzer.py
```

### Custom File Paths
```python
from flexible_weakness_analyzer import WeaknessAnalyzer

analyzer = WeaknessAnalyzer()
report = analyzer.analyze(
    dove_file="path/to/your/dove_scores.json",
    tree_file="path/to/your/eval_tree.json",
    min_questions=3  # Minimum questions per capability area
)
```

## Output Files

### 1. `weakness_profile_summary.json`
- Overall statistics
- Top 20 weakest areas
- Critical weakness areas
- Easy to read and share

### 2. `weakness_profile_detailed.json`
- Complete analysis with all capability areas
- Annotated tree structure with statistics
- Full breakdown by weakness levels
- For detailed analysis

## Weakness Levels

- **Critical** (< 30%): Severe weaknesses requiring immediate attention
- **High** (30-49%): Significant weaknesses needing focus
- **Moderate** (50-69%): Areas for improvement
- **Low** (≥ 70%): Relatively strong areas

## Interpreting Results

### Key Metrics
- **Mean Score**: Average performance in this capability area
- **Standard Deviation**: Consistency of performance
- **Count**: Number of questions analyzed
- **Coverage**: Percentage of questions in this area that were evaluated

### Example Output
```
TOP 10 WEAKEST CAPABILITY AREAS:
 1. Score: 0.156 ± 0.089 | Level: Critical  | Questions:  15
    Analyzing complex mathematical proofs and formal reasoning...

 2. Score: 0.234 ± 0.112 | Level: Critical  | Questions:  23
    Evaluating logical consistency in philosophical arguments...
```

## Troubleshooting

### Common Issues

1. **Empty DOVE scores file**
   - Ensure your file contains valid JSON
   - Check file permissions

2. **Index mismatch**
   - Verify your DOVE indices match the EvalTree leaf indices
   - Check for string vs integer key issues

3. **No capabilities analyzed**
   - Lower the `min_questions` parameter
   - Check if your indices align with the tree structure

### Data Validation
```python
# Check your DOVE scores format
import json
with open('MMLU_DOVE.json', 'r') as f:
    scores = json.load(f)
print(f"Loaded {len(scores)} scores")
print(f"Sample indices: {list(scores.keys())[:5]}")
print(f"Sample scores: {list(scores.values())[:5]}")
```

## Next Steps

1. **Prepare your data**: Ensure DOVE scores are in the correct JSON format
2. **Run analysis**: Execute the weakness analyzer
3. **Review results**: Check the generated summary and detailed reports
4. **Identify patterns**: Look for systematic weaknesses across capability areas
5. **Plan improvements**: Focus on critical and high-weakness areas

## Customization

You can modify the weakness thresholds in the `calculate_node_stats` method:
```python
# Current thresholds
if mean_score < 0.3:      # Critical
elif mean_score < 0.5:    # High  
elif mean_score < 0.7:    # Moderate
else:                     # Low
```

## Contact

If you encounter issues or need help adapting the tool to your specific data format, please provide:
1. Sample of your DOVE scores file structure
2. Any error messages
3. Description of your evaluation methodology