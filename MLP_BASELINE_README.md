# MLP Baseline for xG Prediction

This document describes the MLP (Multi-Layer Perceptron) baseline implementation for Expected Goals (xG) prediction, designed to serve as a performance baseline for comparison with the transformer-based xG models.

## Overview

The MLP baseline provides a simpler alternative to the transformer-based approach for xG prediction. It uses only individual shot events (no sequence context) and a straightforward feedforward neural network architecture.

### Key Differences from Transformer xG Model

| Aspect | Transformer xG | MLP Baseline |
|--------|----------------|--------------|
| **Sequence Length** | 200 events (shot + 199 preceding events) | 1 event (shot only) |
| **Context** | Full match context leading to shot | No contextual information |
| **Architecture** | Transformer with self-attention | Simple feedforward MLP |
| **Masked Features** | Standard xG masks | xG masks + duration + counterpress |
| **Model Complexity** | High (6 transformer layers, attention) | Low (5 fully connected layers) |
| **Training Time** | Longer | Faster |
| **Purpose** | State-of-the-art xG prediction | Baseline for comparison |

## Architecture

### MLP Model Structure
```
Input: [batch_size, embedding_dim=32]
↓
Layer 1: Linear(32 → 32) + ReLU + Dropout
↓
Layer 2: Linear(32 → 32) + ReLU + Dropout  
↓
Layer 3: Linear(32 → 32) + ReLU + Dropout
↓
Layer 4: Linear(32 → 32) + ReLU + Dropout
↓
Layer 5: Linear(32 → 32) + ReLU + Dropout
↓
Output: Linear(32 → 1) + Sigmoid
↓
xG Prediction: [0, 1]
```

### Masked Features

The MLP baseline masks the following features (indices in tokenized vector):

**Common Features:**
- Index 4: `duration` - Shot duration in seconds
- Index 6: `out` - Whether shot went out of play  
- Index 7: `counterpress` - Whether shot was under counterpress

**Shot-Specific Features:**
- Index 72: `shot.end_location[0]` - Shot end location X
- Index 73: `shot.end_location[1]` - Shot end location Y  
- Index 74: `shot.end_location[2]` - Shot end location Z
- Index 75: `shot.aerial_won` - Whether aerial duel was won
- Index 78: `shot.open_goal` - Whether it was an open goal
- Index 79: `shot.statsbomb_xg` - StatsBomb's xG value (ground truth comparison)
- Index 80: `shot.deflected` - Whether shot was deflected
- Index 83: `shot.outcome.id` - Shot outcome (goal/no goal)

The additional masking of `duration` and `counterpress` (compared to the transformer model) tests whether these features provide useful signal or if removing them improves generalization.

## Usage

### 1. Training the MLP Baseline

```bash
# Basic training
python main.py \
    --task mlp_baseline \
    --model-type regression \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 1 \
    --batch-size 64 \
    --epochs 30 \
    --learning-rate 1e-3

# With custom parameters
python main.py \
    --task mlp_baseline \
    --model-type regression \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 1 \
    --checkpoint-dir checkpoints/mlp_baseline \
    --artifacts-dir artifacts/mlp_baseline \
    --batch-size 128 \
    --epochs 50 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4
```

### 2. Evaluating the MLP Baseline

```bash
# Standalone evaluation
python mlp_baseline_evaluation.py \
    --mlp-model-path checkpoints/mlp_baseline/best_model.pth \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --num-samples 20

# With transformer comparison
python mlp_baseline_evaluation.py \
    --mlp-model-path checkpoints/mlp_baseline/best_model.pth \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --transformer-model-path checkpoints/xg_prediction/best_model.pth \
    --num-samples 50
```

### 3. Using the Example Script

```bash
# Train and evaluate in one command
python example_mlp_baseline.py \
    --mode both \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --transformer-model-path /path/to/transformer.pth

# Just train
python example_mlp_baseline.py \
    --mode train \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth

# Just evaluate with comparison
python example_mlp_baseline.py \
    --mode evaluate \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --transformer-model-path /path/to/transformer.pth
```

## Implementation Files

### Core Components

1. **`sequence_classifier/sequence_transformer.py`**
   - Added `MLPBaseline` class
   - 5-layer feedforward network with ReLU activations and dropout
   - Sigmoid output for [0,1] xG values

2. **`sequence_classifier/dataset.py`**
   - Added `mlp_baseline` task registration
   - Sampling logic for single shot events (sequence_length=1)
   - Data loading functions for MLP baseline

3. **`sequence_classifier/main_transformer.py`**
   - Updated to support MLP baseline task
   - Added mask_list for MLP baseline (includes duration and counterpress)
   - Model creation logic for MLPBaseline

4. **`mlp_baseline_evaluation.py`**
   - Comprehensive evaluation script
   - Side-by-side comparison with transformer models
   - Detailed shot-by-shot analysis
   - Performance metrics and visualizations

5. **`example_mlp_baseline.py`**
   - User-friendly script for training and evaluation
   - Automated workflow management
   - Clear documentation and examples

## Expected Performance

### Baseline Characteristics

The MLP baseline is intentionally designed to be simpler and less capable than the transformer model:

- **Lower Accuracy**: Expected to perform worse than transformer due to lack of sequential context
- **Faster Training**: Should train significantly faster due to simpler architecture
- **Less Overfitting**: Simpler model may generalize better on small datasets
- **Interpretable**: More straightforward to analyze and debug

### Key Metrics to Monitor

1. **MSE vs StatsBomb xG**: How well does the model approximate professional xG values?
2. **Relative Performance Score**: How often does the model outperform StatsBomb on actual outcomes?
3. **MLP vs Transformer Gap**: What's the performance difference between simple and complex approaches?
4. **Training Efficiency**: Training time and convergence speed comparison

## Use Cases

### 1. Baseline Establishment
- Provides a performance floor for more complex models
- Helps quantify the value added by transformer architecture
- Validates that improvements come from model design, not just more data

### 2. Ablation Studies
- Tests the importance of sequential context in xG prediction
- Evaluates the impact of masking duration and counterpress features
- Helps identify which features drive xG predictions

### 3. Production Considerations
- Faster inference for real-time applications
- Lower computational requirements
- Simpler deployment and maintenance

### 4. Model Comparison
- Direct comparison with StatsBomb xG values
- Benchmark for evaluating new architectures
- Educational tool for understanding xG modeling approaches

## Tips for Best Results

### Training
- Use a higher learning rate (1e-3) than transformer models
- Consider shorter training (20-30 epochs) to avoid overfitting
- Monitor validation loss carefully due to simpler architecture
- Experiment with different hidden layer sizes

### Evaluation  
- Always evaluate on test set matches only (matching transformer evaluation)
- Use balanced samples (equal goals and non-goals) for fair comparison
- Focus on relative performance scores rather than just MSE
- Compare against both StatsBomb xG and transformer predictions

### Analysis
- Look for cases where MLP outperforms transformer (may indicate over-complexity)
- Analyze failure modes to understand limitations
- Use results to motivate transformer architecture improvements
- Consider ensemble approaches combining both models

## Future Enhancements

1. **Feature Engineering**: Add hand-crafted features that might help the MLP
2. **Architecture Variants**: Experiment with different layer sizes and depths
3. **Regularization**: Try different dropout rates and L2 penalties
4. **Ensemble Methods**: Combine MLP and transformer predictions
5. **Specialized Evaluation**: Focus on specific shot types or game situations

This MLP baseline provides a solid foundation for understanding the performance gains achieved by the more sophisticated transformer-based approach to xG prediction.