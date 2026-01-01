# User Manual

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

Required files:
- Match event CSV files (tokenized StatsBomb data)
- Pre-trained autoencoder model checkpoint
- GPU recommended for training (CPU supported)

## Basic Usage

### Training an xG Model

```bash
python main.py \
    --task xg_prediction \
    --model-type regression \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 200 \
    --batch-size 64 \
    --epochs 30 \
    --learning-rate 1e-4
```

### Evaluating a Trained Model

```bash
python xg_evaluation.py \
    --model-path checkpoints/best_model.pth \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --num-samples 20
```

### Training an MLP Baseline

```bash
python main.py \
    --task mlp_baseline \
    --model-type regression \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 1 \
    --learning-rate 1e-3
```

### Training a Team Dominance Classification Model

```bash
python main.py \
    --task dominating_team_classification \
    --model-type classification \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 50 \
    --batch-size 64 \
    --epochs 30 \
    --learning-rate 1e-4
```

### Training a Team Dominance Regression Model

```bash
python main.py \
    --task dominating_team_regression \
    --model-type regression \
    --data-dir /path/to/match_csv \
    --encoder-path /path/to/encoder.pth \
    --sequence-length 50 \
    --batch-size 64 \
    --epochs 30 \
    --learning-rate 1e-4
```

## Available Parameters

### Task Parameters

- `--task` (required): Task to train - `xg_prediction`, `temporal_ordering`, `dominating_team_classification`, `dominating_team_regression`, or `mlp_baseline`
- `--model-type`: Model type - `classification` or `regression` (default: `regression`)
  - Use `classification` for: `temporal_ordering`, `dominating_team_classification`
  - Use `regression` for: `xg_prediction`, `dominating_team_regression`, `mlp_baseline`

### Data and Model Paths

- `--data-dir`: Directory containing match CSV files (default: `../match_csv`)
- `--encoder-path` (required): Path to pretrained encoder model
- `--cache-dir`: Directory to cache embeddings (default: `cache`)
- `--checkpoint-dir`: Directory to save model checkpoints (default: `checkpoints`)
- `--artifacts-dir`: Directory to save output artifacts (default: `artifacts`)

### Dataset Parameters

- `--sequence-length`: Number of events in each sequence (default: 10)
- `--min-gap`: Minimum events between sequences (default: 1)
- `--max-gap`: Maximum events between sequences (default: None)
- `--train-ratio`: Proportion of matches for training (default: 0.8)
- `--val-ratio`: Proportion of matches for validation (default: 0.1)
- `--max-samples-per-match`: Maximum samples per match (default: 10000)
- `--max-samples-total`: Maximum total samples (default: 1000000)

### Transformer Model Parameters

- `--transformer-heads`: Number of attention heads (default: 8)
- `--transformer-layers`: Number of transformer layers (default: 6)
- `--transformer-dim-feedforward`: Dimension of feedforward network (default: 2048)
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters

- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 30)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--patience`: Early stopping patience (default: 5)
- `--clip-grad-norm`: Gradient clipping norm (default: 1.0)
- `--num-workers`: Number of data loader workers (default: 4)

### Utility Flags

- `--seed`: Random seed for reproducibility (default: 42)
- `--force-recompute`: Force recomputation of embeddings
- `--skip-training`: Skip training (evaluation only)
- `--skip-evaluation`: Skip evaluation (training only)
- `--cpu`: Force CPU usage even if GPU is available
- `--debug`: Enable debug logging
- `--quiet`: Disable progress bars

