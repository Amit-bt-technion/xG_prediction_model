# Football Task Models

This project trains transformer-based models to understand football match dynamics using StatsBomb's open event data. The pipeline processes raw event data through an autoencoder for dimensionality reduction, then trains task-specific models for analyzing game situations like expected goals (xG) prediction and temporal event ordering.

## Project Overview

The system follows a multi-stage pipeline:

1. **Event Tokenization**: Raw StatsBomb JSON events are converted into normalized 128-dimensional feature vectors
2. **Embedding**: An autoencoder compresses these vectors into 32-dimensional embeddings
3. **Task-Specific Training**: Transformer models learn from sequences of embedded events to perform specific tasks
4. **Evaluation**: Specialized evaluation tools compare model predictions against StatsBomb benchmarks

The architecture uses sequence transformers with positional encoding to capture the temporal dependencies between football events, enabling the model to understand context and game flow.

## Core Files

### Entry Points

**`main.py`**  
Primary training script with command-line interface for all task-specific models. Handles argument parsing and delegates to the transformer training pipeline. Supports tasks like xG prediction, temporal ordering, and MLP baselines. Configure data paths, model hyperparameters, training parameters, and evaluation settings through CLI arguments.

**`example_xg_evaluation.py`**  
Quick-start example demonstrating how to evaluate a trained xG model. Checks for required files, loads a trained model checkpoint, and runs detailed evaluation with sample predictions. Useful template for setting up your own evaluation workflows.

### Model Architectures

**`sequence_classifier/sequence_transformer.py`**  
Core transformer model implementations:
- `EventSequenceTransformer`: Classification model with CLS token for temporal ordering tasks
- `ContinuousValueTransformer`: Regression model for continuous predictions (xG values)
- `MLPBaseline`: Simple feedforward baseline for comparison (no sequential context)
- `PositionalEncoding`: Injects temporal position information into event embeddings

All models use multi-head self-attention to capture relationships between events in a sequence.

**`utils/event_autoencoder.py`**  
Defines the `EventAutoencoder` class - a 5-layer encoder-decoder network that compresses 128-dimensional tokenized events into 32-dimensional latent embeddings. The encoder must be pre-trained on event reconstruction before being used in the main pipeline. Only the encoder portion is used during task-specific training.

### Data Processing

**`tokenized_eventpy.py`**  
Event tokenization configuration defining how raw StatsBomb events are converted to feature vectors. Contains parsers for 30+ event types (shots, passes, dribbles, etc.) with both common features (location, timing, pressure) and event-specific features. Maps categorical values to normalized ranges and handles missing data. Output is a 128-dimensional vector per event.

**`sequence_classifier/dataset.py`**  
Custom PyTorch Dataset implementation with task-specific sampling logic:
- `EventSequenceDataset`: Main dataset class supporting multiple tasks via a registry pattern
- Task-specific sampling functions for different prediction problems
- Handles sequence windowing, gap constraints, and label extraction
- Manages efficient data loading with pre-computed embeddings cached by match

Tasks are registered using decorators and can define custom sampling and labeling strategies.

**`sequence_classifier/preprocessing.py`**  
Handles loading, embedding, and caching of match data:
- `load_and_embed_matches()`: Loads CSV files, applies autoencoder, caches results
- Supports task-specific masking (hiding features the model shouldn't see)
- Implements intelligent caching to avoid recomputing embeddings
- Processes data match-by-match for memory efficiency

Masking is crucial for tasks like xG prediction where the model shouldn't see the shot outcome during training.

**`utils/data_loading.py`**  
Basic utilities for loading and cleaning match CSV files. Combines multiple match files into a single DataFrame with match IDs. Performs minimal cleanup of unnecessary columns. Simple but essential for the data pipeline.

### Training and Evaluation

**`sequence_classifier/training.py`**  
Core training loop utilities:
- `train_epoch()`: Single epoch training with gradient clipping
- `validate()`: Validation loop with metrics
- `train_model()`: Full training orchestration with early stopping, checkpointing, and learning rate scheduling
- `plot_training_history()`: Visualization of loss and accuracy curves

Handles both classification and regression tasks with appropriate loss functions.

**`sequence_classifier/regression_training.py`**  
Specialized training functions for regression tasks (xG prediction). Similar structure to classification training but uses MSE loss and different evaluation metrics. Tracks RMSE and MAE instead of accuracy.

**`sequence_classifier/main_transformer.py`**  
Main training orchestration that ties everything together:
- Creates task-specific data loaders with proper train/val/test splits
- Initializes appropriate model architecture based on task
- Sets up loss functions and optimizers
- Manages checkpointing and artifact saving
- Handles both training and evaluation phases

This is the main entry point called by `main.py`.

### Evaluation Scripts

**`xg_evaluation.py`**  
Comprehensive evaluation framework for xG models:
- Samples balanced shots (goals/non-goals) from test set only
- Compares model predictions against StatsBomb xG values
- Calculates relative performance scores
- Displays detailed shot characteristics and context
- Ensures proper test set isolation to prevent data leakage

Uses the same train/test split as training to guarantee valid evaluation.

**`mlp_baseline_evaluation.py`**  
Similar evaluation framework for MLP baseline models with additional comparison features:
- Evaluates simple feedforward models that see only individual shots
- Can compare MLP vs Transformer performance side-by-side
- Quantifies the value added by sequential context
- Helps establish performance baselines and validate that improvements come from architecture

Useful for ablation studies and understanding model contributions.

### Visualization and Utilities

**`utils/visualization.py`**  
Plotting and visualization utilities for analysis and presentation (implementation details not shown).

## Directory Structure

```
.
├── cache/                    # Cached embeddings (auto-generated)
├── checkpoints/              # Saved model checkpoints
├── artifacts/                # Training plots and logs
├── sequence_classifier/      # Main model package
│   ├── dataset.py           # Data loading and sampling
│   ├── preprocessing.py     # Embedding and caching
│   ├── sequence_transformer.py  # Model architectures
│   ├── training.py          # Training loops
│   ├── regression_training.py   # Regression-specific training
│   └── main_transformer.py  # Training orchestration
├── utils/                    # Utility modules
│   ├── data_loading.py      # CSV loading
│   ├── event_autoencoder.py # Autoencoder model
│   └── visualization.py     # Plotting functions
├── main.py                   # Primary training script
├── xg_evaluation.py         # xG model evaluation
├── mlp_baseline_evaluation.py  # Baseline evaluation
├── example_xg_evaluation.py # Evaluation example
├── tokenized_eventpy.py     # Event tokenization config
└── requirements.txt         # Python dependencies
```

## Available Tasks

The task registry in `dataset.py` defines available prediction tasks:
- **`xg_prediction`**: Predict expected goals for shot events using preceding context
- **`temporal_ordering`**: Binary classification of whether event sequences are in chronological order
- **`dominating_team_classification`**: Binary classification to predict which team dominated a sequence based on possession time
- **`dominating_team_regression`**: Regression to predict team 0's possession percentage (0-1 range) over a sequence
- **`mlp_baseline`**: xG prediction without sequential context (single-event baseline)

Additional tasks can be added by implementing sampling logic and registering with the task registry.

## Model Comparison

The project supports direct comparison between:
- **Transformer models**: Use full event sequences for context-aware predictions
- **MLP baselines**: Use only individual events without context
- **StatsBomb xG**: Industry-standard benchmark values

This enables quantifying the value of sequential modeling and validating architectural choices.

## Additional Documentation

- `USER_MANUAL.md`: User guide with prerequisites, usage examples, and parameter documentation
- `MLP_BASELINE_README.md`: Detailed guide for MLP baseline implementation and usage
- `XG_EVALUATION_README.md`: Comprehensive evaluation methodology and metrics explanation
- `StatsBomb Open Data Specification v1.1.pdf`: Event data format documentation

## Data Source

This project uses StatsBomb's open event data. The events must be pre-processed into CSV format with the tokenization scheme defined in `tokenized_eventpy.py`.

## License

See `LICENSE` file for details.