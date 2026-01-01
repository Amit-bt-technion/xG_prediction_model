# xG Model Evaluation

This document describes the comprehensive evaluation functionality for xG (expected goals) prediction models.

## ⚠️ CRITICAL: Test Set Isolation

**This evaluation script ensures ZERO data leakage by:**
- Using the exact same train/test split as the training process
- Only evaluating samples from the test set (never seen during training)
- Validating that all samples come from test matches only
- Requiring matching `train_ratio`, `val_ratio`, and `seed` parameters

**You MUST use the same split parameters as training to ensure valid evaluation!**

## Overview

The xG evaluation script (`xg_evaluation.py`) provides detailed analysis of trained xG models by:

1. **Sampling balanced shots**: 10 goals and 10 non-goals (or custom amounts) **from test set only**
2. **Model predictions**: Raw 0-1 probability values from your trained model
3. **StatsBomb comparison**: Original StatsBomb xG values for reference
4. **📊 NEW: Relative performance analysis**: Sophisticated metric comparing model vs StatsBomb performance
5. **Comprehensive shot data**: Detailed information about each shot scenario

## Features

### 📊 Balanced Sampling from Test Set Only
- **CRITICAL**: Only samples from test set matches to ensure no data leakage
- Applies same train/test split as training (80% train, 10% val, 10% test by default)
- Automatically finds and samples equal numbers of goal and non-goal shots
- Ensures sufficient sequence history (200 events before each shot)
- Randomized selection for unbiased evaluation within test set

### 🎯 Prediction Comparison
- **Model xG**: Raw continuous output from your trained regression model
- **StatsBomb xG**: Original StatsBomb expected goals value
- **Difference**: Absolute difference between model and StatsBomb predictions
- **📊 Relative Performance Score**: Measures how much better/worse the model is vs StatsBomb relative to actual outcome

### ⚽ Comprehensive Shot Details
For each shot, displays:

**Location & Timing:**
- Shot location coordinates (x, y)
- End location (x, y, z) including height
- Match period and minute
- Shot duration

**Shot Characteristics:**
- First time shot
- Aerial won
- Open goal opportunity
- Deflected shot
- Follows dribble

**Context:**
- Under pressure
- Counterpress situation
- Team ID and possession
- Player position

### 📈 Summary Statistics
- Total samples evaluated
- Goal vs non-goal breakdown
- **📊 Relative Performance Analysis**: How often model beats/matches/loses to StatsBomb XG
- **Average Relative Score**: Overall performance comparison metric
- MSE and RMSE against StatsBomb xG
- Overall evaluation summary

## Usage

### Command Line Interface

```bash
python3 xg_evaluation.py \
  --model-path checkpoints/best_model.pth \
  --data-dir /path/to/match/csv/files \
  --encoder-path /path/to/encoder.pth \
  --num-samples 20 \
  --sequence-length 200 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

### Programmatic Usage

```python
from xg_evaluation import evaluate_xg_model_detailed

evaluate_xg_model_detailed(
    model_path="checkpoints/best_model.pth",
    data_dir="/path/to/match/data",
    encoder_path="/path/to/encoder.pth",
    num_samples=20,  # 10 goals + 10 non-goals
    sequence_length=200,
    train_ratio=0.8,  # Must match training split
    val_ratio=0.1,    # Must match training split
    seed=42           # Must match training seed
)
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | str | Required | Path to trained xG model checkpoint |
| `data_dir` | str | Required | Directory containing match CSV files |
| `encoder_path` | str | Required | Path to pretrained event encoder |
| `cache_dir` | str | "cache" | Cache directory for embeddings |
| `sequence_length` | int | 200 | Number of events in sequence |
| `num_samples` | int | 20 | Total samples (should be even) |
| `train_ratio` | float | 0.8 | **CRITICAL**: Must match training split ratio |
| `val_ratio` | float | 0.1 | **CRITICAL**: Must match training split ratio |
| `seed` | int | 42 | **CRITICAL**: Must match training seed for reproducible split |

## Prerequisites

### 1. Trained xG Model
First train an xG prediction model:

```bash
python3 main.py \
  --task=xg_prediction \
  --model-type=regression \
  --sequence-length=200 \
  --data-dir=/path/to/data \
  --encoder-path=/path/to/encoder.pth \
  --epochs=30
```

### 2. Required Files
- **Model checkpoint**: `checkpoints/best_model.pth` or `checkpoints/final_model.pth`
- **Match data**: CSV files with StatsBomb event data
- **Encoder**: Pretrained event autoencoder model

### 3. Dependencies
Ensure you have the required packages (from `requirements.txt`):
```
torch
numpy
pandas
```

## Example Output

```
====================================================================================================
XG MODEL EVALUATION - 20 SAMPLES
====================================================================================================

SAMPLE 1/20

================================================================================
SHOT OUTCOME: GOAL
================================================================================

🎯 PREDICTIONS:
   Model xG:     0.8234
   StatsBomb xG: 0.7856
   Difference:   0.0378

📊 PERFORMANCE COMPARISON:
   Relative Score: +0.647 🟢
   Interpretation: Model BETTER than StatsBomb

📍 LOCATION & TIMING:
   Shot Location:  (102.3, 34.5)
   End Location:   (120.0, 40.2, 1.2m)
   Period:         2 
   Minute:         67.3
   Duration:       0.15s

⚽ SHOT CHARACTERISTICS:
   First Time:     Yes
   Aerial Won:     No
   Open Goal:      No
   Deflected:      No
   Follows Dribble: No

🔥 CONTEXT:
   Under Pressure: Yes
   Counterpress:   No
   Team ID:        0
   Position ID:    23

...

====================================================================================================
EVALUATION SUMMARY
====================================================================================================
Total Samples:     20
Goals:             10
Non-Goals:         10

📊 RELATIVE PERFORMANCE vs StatsBomb:
   Average Score:   +0.1234
   Better:          12/20 (60.0%) 🟢
   Similar:         5/20 (25.0%) 🟡
   Worse:           3/20 (15.0%) 🔴

📈 TRADITIONAL METRICS:
   Model vs StatsBomb MSE:  0.012456
   Model vs StatsBomb RMSE: 0.111600
====================================================================================================
```

## Technical Details

### Data Flow
1. **Load Data**: Uses same preprocessing as training with masking for xG prediction
2. **Sample Shots**: Finds shots with sufficient sequence history
3. **Model Inference**: Runs sequences through trained model for predictions
4. **Data Extraction**: Extracts original StatsBomb data from unmasked events
5. **Display**: Formats and presents comprehensive shot information

### Masking
The evaluation automatically applies the same field masking used during training:
- `out` (index 6)
- `shot.end_location[0,1,2]` (indices 72, 73, 74)
- `shot.aerial_won` (index 75)
- `shot.open_goal` (index 78) 
- `shot.statsbomb_xg` (index 79)
- `shot.deflected` (index 80)
- `shot.outcome.id` (index 83)

This ensures the model doesn't have access to the target information during prediction.

### Relative Performance Score

**NEW METRIC**: The evaluation now includes a sophisticated metric that measures how much better (or worse) your model performs compared to StatsBomb xG relative to the actual outcome.

#### Formula
```
Relative Score = (StatsBomb_Error - Model_Error) / max(StatsBomb_Error, Model_Error, ε)

Where:
- Model_Error = |model_xg - actual_outcome|
- StatsBomb_Error = |statsbomb_xg - actual_outcome|
- ε = small constant (0.01 default) to avoid division by zero
```

#### Interpretation
- **Score > 0.1**: Model BETTER than StatsBomb 🟢
- **Score < -0.1**: Model WORSE than StatsBomb 🔴  
- **-0.1 ≤ Score ≤ 0.1**: Model SIMILAR to StatsBomb 🟡

#### Examples
- **Goal (actual=1)**: Model=0.88, StatsBomb=0.5 → Score=+0.76 🟢 (Model much closer to 1)
- **No Goal (actual=0)**: Model=0.17, StatsBomb=0.05 → Score=-0.71 🔴 (StatsBomb much closer to 0)
- **Perfect Model**: Model=1.0, StatsBomb=0.7, Actual=1 → Score=+1.0 🟢 (Maximum advantage)

This metric provides intuitive insight into when your model makes better predictions than the industry standard StatsBomb xG.

### Data Decoding
The script includes comprehensive decoding of normalized event vectors back to human-readable values:
- Location coordinates denormalized to pitch dimensions (120x80)
- Temporal features converted to actual minutes/seconds
- Categorical features mapped to meaningful labels
- Height information converted to meters

## Model Architecture Assumptions

The evaluation assumes:
- **Model Type**: `ContinuousValueTransformer` (regression model)
- **Input**: Embedded sequences of length 200
- **Output**: Single continuous value (xG prediction)
- **Embedding Dimension**: 32 (matching encoder output)

If your model has different architecture, update the model creation in `evaluate_xg_model_detailed()`.

## Troubleshooting

### Common Issues

1. **"No shots found"**: Ensure your data contains shot events and has sufficient sequence history
2. **Model loading errors**: Check that model path is correct and architecture matches
3. **Memory issues**: Reduce `num_samples` or `sequence_length` for large datasets
4. **Cache conflicts**: Clear cache directory if switching between different tasks

### Performance Tips

- Use cached embeddings to speed up repeated evaluations
- Reduce `num_samples` for quick testing
- Ensure sufficient disk space for cache files
- Use GPU if available for faster inference

## Integration

The evaluation script integrates seamlessly with the existing xG training pipeline:
- Uses same preprocessing and masking logic
- Compatible with trained model checkpoints
- Leverages existing dataset and embedding infrastructure
- Maintains consistency with training data flow

This ensures that evaluation results accurately reflect model performance on the same type of data used during training. 