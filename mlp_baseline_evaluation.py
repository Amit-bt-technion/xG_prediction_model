"""
mlp_baseline_evaluation.py
--------------------------
Specialized evaluation functionality for MLP baseline models, designed to compare
against the transformer-based xG prediction models.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from sequence_classifier.dataset import EventSequenceDataset, is_shot_event, extract_goal_label_from_shot
from sequence_classifier.preprocessing import load_and_embed_matches
from sequence_classifier.sequence_transformer import MLPBaseline, ContinuousValueTransformer
from xg_evaluation import sample_balanced_shots


def sample_balanced_shots_for_mlp(
    events_dict: Dict[str, np.ndarray],
    target_goals: int = 10,
    target_no_goals: int = 10
) -> List[Tuple[str, int, int]]:
    """
    Sample a balanced set of shots (goals and non-goals) for MLP evaluation.
    
    Args:
        events_dict: Dictionary of match events
        target_goals: Number of goal shots to sample
        target_no_goals: Number of non-goal shots to sample
        
    Returns:
        List of (match_id, shot_index, goal_label) tuples
    """
    goal_samples = []
    no_goal_samples = []
    
    # Collect all valid shots
    for match_id, match_events in events_dict.items():
        num_events = len(match_events)
        
        # Look for all shot events
        for i in range(num_events):
            event_vector = match_events[i]
            
            if is_shot_event(event_vector):
                goal_label = extract_goal_label_from_shot(event_vector)
                
                if goal_label == 1 and len(goal_samples) < target_goals:
                    goal_samples.append((match_id, i, goal_label))
                elif goal_label == 0 and len(no_goal_samples) < target_no_goals:
                    no_goal_samples.append((match_id, i, goal_label))
                
                # Stop if we have enough samples
                if len(goal_samples) >= target_goals and len(no_goal_samples) >= target_no_goals:
                    break
        
        if len(goal_samples) >= target_goals and len(no_goal_samples) >= target_no_goals:
            break
    
    # Shuffle and combine
    random.shuffle(goal_samples)
    random.shuffle(no_goal_samples)
    
    # Take the required number of each
    selected_goals = goal_samples[:target_goals]
    selected_no_goals = no_goal_samples[:target_no_goals]
    
    # Combine and shuffle the final list
    all_samples = selected_goals + selected_no_goals
    random.shuffle(all_samples)
    
    return all_samples


def calculate_relative_performance_score(model_xg: float, statsbomb_xg: float, actual_outcome: int) -> float:
    """
    Calculate a relative performance score comparing model vs StatsBomb xG.
    
    Args:
        model_xg: Model's xG prediction (0-1)
        statsbomb_xg: StatsBomb's xG value (0-1)  
        actual_outcome: Actual outcome (0 = no goal, 1 = goal)
        
    Returns:
        float: Relative performance score
            > 0: Model is better than StatsBomb
            < 0: Model is worse than StatsBomb  
            ≈ 0: Similar performance
    """
    # Calculate absolute errors
    model_error = abs(model_xg - actual_outcome)
    statsbomb_error = abs(statsbomb_xg - actual_outcome)
    
    # Small epsilon to avoid division by zero
    epsilon = 0.01
    
    # Calculate relative performance score
    denominator = max(statsbomb_error, model_error, epsilon)
    score = (statsbomb_error - model_error) / denominator
    
    return score


def decode_shot_event_data(event_vector: np.ndarray) -> Dict:
    """
    Decode shot event data into human-readable format.
    
    Args:
        event_vector: The raw event vector from events_dict
        
    Returns:
        Dictionary containing decoded shot information
    """
    # Based on tokenized_eventpy.py structure
    common_data = {
        'type_id': event_vector[0],
        'play_pattern_id': event_vector[1],
        'location_x': event_vector[2] * 120,
        'location_y': event_vector[3] * 80,
        'duration': event_vector[4] * 3,
        'under_pressure': bool(event_vector[5] > 0.5),
        'out': bool(event_vector[6] > 0.5),
        'counterpress': bool(event_vector[7] > 0.5),
        'period': int(event_vector[8] * 5) + 1,
        'second': int(event_vector[9] * 60),
        'position_id': int(event_vector[10] * 25) + 1,
        'minute': event_vector[11] * 60,
        'team_id': int(event_vector[12]),
        'possession_team_id': int(event_vector[13]),
        'player_designated_position': event_vector[14],
    }
    
    # Shot-specific features (starting at index 71):
    shot_data = {
        'shot_type_id': event_vector[71],
        'end_location_x': event_vector[72] * 120,
        'end_location_y': event_vector[73] * 80,
        'end_location_z': event_vector[74] * 5,
        'aerial_won': bool(event_vector[75] > 0.5),
        'follows_dribble': bool(event_vector[76] > 0.5),
        'first_time': bool(event_vector[77] > 0.5),
        'open_goal': bool(event_vector[78] > 0.5),
        'statsbomb_xg': event_vector[79],
        'deflected': bool(event_vector[80] > 0.5),
        'technique_id': event_vector[81],
        'body_part_id': event_vector[82],
        'outcome_id': event_vector[83],
    }
    
    return {**common_data, **shot_data}


def format_shot_display_with_comparison(
    shot_data: Dict, 
    mlp_prediction: float, 
    transformer_prediction: Optional[float],
    goal_label: int, 
    mlp_relative_score: float,
    transformer_relative_score: Optional[float] = None
) -> str:
    """
    Format shot data for comprehensive display comparing MLP baseline vs Transformer.
    
    Args:
        shot_data: Dictionary of decoded shot information
        mlp_prediction: MLP baseline prediction (0-1)
        transformer_prediction: Transformer prediction (0-1), if available
        goal_label: Actual goal outcome (0 or 1)
        mlp_relative_score: Score comparing MLP vs StatsBomb performance
        transformer_relative_score: Score comparing Transformer vs StatsBomb performance
        
    Returns:
        Formatted string for display
    """
    outcome_text = "GOAL" if goal_label == 1 else "NO GOAL"
    
    display = f"""
{'='*80}
SHOT OUTCOME: {outcome_text}
{'='*80}

🎯 PREDICTIONS:
   MLP Baseline xG:  {mlp_prediction:.4f}
   StatsBomb xG:     {shot_data['statsbomb_xg']:.4f}"""
    
    if transformer_prediction is not None:
        display += f"""
   Transformer xG:   {transformer_prediction:.4f}"""
    
    display += f"""
   
📊 PERFORMANCE COMPARISON:
   MLP vs StatsBomb:     {mlp_relative_score:+.3f} {'🟢' if mlp_relative_score > 0.1 else '🔴' if mlp_relative_score < -0.1 else '🟡'}"""
    
    if transformer_relative_score is not None:
        display += f"""
   Transformer vs StatsBomb: {transformer_relative_score:+.3f} {'🟢' if transformer_relative_score > 0.1 else '🔴' if transformer_relative_score < -0.1 else '🟡'}
   MLP vs Transformer:   {mlp_prediction - transformer_prediction:+.4f} {'🟢 MLP Better' if mlp_prediction > transformer_prediction else '🔴 Transformer Better' if mlp_prediction < transformer_prediction else '🟡 Similar'}"""
    
    display += f"""

📍 LOCATION & TIMING:
   Shot Location:  ({shot_data['location_x']:.1f}, {shot_data['location_y']:.1f})
   Period:         {shot_data['period']} 
   Minute:         {shot_data['minute']:.1f}
   Duration:       {shot_data['duration']:.2f}s

⚽ SHOT CHARACTERISTICS:
   First Time:     {'Yes' if shot_data['first_time'] else 'No'}
   Aerial Won:     {'Yes' if shot_data['aerial_won'] else 'No'}
   Open Goal:      {'Yes' if shot_data['open_goal'] else 'No'}
   Deflected:      {'Yes' if shot_data['deflected'] else 'No'}
   Follows Dribble: {'Yes' if shot_data['follows_dribble'] else 'No'}

🔥 CONTEXT:
   Under Pressure: {'Yes' if shot_data['under_pressure'] else 'No'}
   Counterpress:   {'Yes' if shot_data['counterpress'] else 'No'}
   Team ID:        {shot_data['team_id']}
   Position ID:    {shot_data['position_id']}

"""
    return display


def evaluate_mlp_baseline_detailed(
    mlp_model_path: str,
    data_dir: str,
    encoder_path: str,
    cache_dir: str = "cache",
    device: Optional[torch.device] = None,
    num_samples: int = 20,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    transformer_model_path: Optional[str] = None
) -> None:
    """
    Detailed evaluation of MLP baseline model with comparison to transformer model.
    
    Args:
        mlp_model_path: Path to trained MLP baseline model checkpoint
        data_dir: Directory containing match CSV files
        encoder_path: Path to pretrained encoder
        cache_dir: Cache directory for embeddings
        device: Device to run evaluation on
        num_samples: Total number of samples to evaluate (should be even)
        train_ratio: Proportion of matches used for training (to exclude from evaluation)
        val_ratio: Proportion of matches used for validation (to exclude from evaluation)
        seed: Random seed for reproducible train/test split
        transformer_model_path: Optional path to transformer model for comparison
    """
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure even number of samples for balanced evaluation
    if num_samples % 2 != 0:
        num_samples += 1
        logger.warning(f"Adjusted num_samples to {num_samples} for balanced evaluation")
    
    target_goals = num_samples // 2
    target_no_goals = num_samples // 2
    
    logger.info(f"Starting detailed MLP baseline evaluation with {target_goals} goals and {target_no_goals} non-goals")
    
    # Load data with masking for MLP baseline
    mask_list = [4, 6, 7, 72, 73, 74, 75, 78, 79, 80, 83]  # duration, out, counterpress, end_location[0,1,2], aerial_won, open_goal, statsbomb_xg, deflected, outcome.id
    
    logger.info("Loading and embedding match data...")
    events_dict, embeddings_dict = load_and_embed_matches(
        csv_root_dir=data_dir,
        encoder_model_path=encoder_path,
        cache_dir=cache_dir,
        device=device,
        batch_size=32,
        force_recompute=False,
        verbose=True,
        mask_list=mask_list,
        task="mlp_baseline"
    )
    
    # Apply same train/test split as training to get test set matches only
    logger.info("Applying train/test split to ensure evaluation uses only test set...")
    match_ids = list(events_dict.keys())
    
    # Set random seed for reproducible split (same as training)
    random_state = random.getstate()
    random.seed(seed)
    random.shuffle(match_ids)
    random.setstate(random_state)
    
    # Split match IDs into train, validation, and test sets
    num_matches = len(match_ids)
    train_size = int(train_ratio * num_matches)
    val_size = int(val_ratio * num_matches)
    
    test_match_ids = match_ids[train_size + val_size:]
    logger.info(f"Using {len(test_match_ids)} test matches out of {num_matches} total matches")
    
    # Filter events_dict to only include test matches
    test_events_dict = {match_id: events_dict[match_id] for match_id in test_match_ids}
    
    # Sample balanced shots from test set only
    logger.info("Sampling balanced shots from TEST SET ONLY...")
    shot_samples = sample_balanced_shots_for_mlp(
        events_dict=test_events_dict,
        target_goals=target_goals,
        target_no_goals=target_no_goals
    )
    
    if len(shot_samples) < num_samples:
        logger.warning(f"Only found {len(shot_samples)} valid shots, expected {num_samples}")
    
    # Load MLP model
    logger.info(f"Loading MLP baseline model from {mlp_model_path}")
    mlp_model = MLPBaseline(
        embedding_dim=32,
        hidden_dims=[32, 32, 32, 32, 32],
        dropout=0.1
    )
    
    checkpoint = torch.load(mlp_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        mlp_model.load_state_dict(checkpoint)
    
    mlp_model = mlp_model.to(device)
    mlp_model.eval()
    
    # Optionally load transformer model for comparison
    transformer_model = None
    if transformer_model_path:
        logger.info(f"Loading transformer model from {transformer_model_path} for comparison")
        # We'll need to load transformer embeddings with different mask_list
        xg_mask_list = [4, 6, 7, 72, 73, 74, 75, 78, 79, 80, 83]
        _, transformer_embeddings_dict = load_and_embed_matches(
            csv_root_dir=data_dir,
            encoder_model_path=encoder_path,
            cache_dir=cache_dir,
            device=device,
            batch_size=32,
            force_recompute=False,
            verbose=True,
            mask_list=xg_mask_list,
            task="xg_prediction"
        )
        
        transformer_model = ContinuousValueTransformer(
            embedding_dim=32,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        transformer_checkpoint = torch.load(transformer_model_path, map_location=device)
        if 'model_state_dict' in transformer_checkpoint:
            transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        else:
            transformer_model.load_state_dict(transformer_checkpoint)
        
        transformer_model = transformer_model.to(device)
        transformer_model.eval()

        shot_samples = sample_balanced_shots(
            events_dict=test_events_dict,
            target_goals=target_goals,
            target_no_goals=target_no_goals,
            sequence_length=50
        )

    # Evaluate each sample
    logger.info("Evaluating samples...")
    print("\n" + "="*100)
    print(f"MLP BASELINE EVALUATION - {len(shot_samples)} SAMPLES")
    if transformer_model:
        print("(WITH TRANSFORMER COMPARISON)")
    print("="*100)
    
    goal_count = 0
    no_goal_count = 0
    total_mlp_mse = 0.0
    total_mlp_relative_score = 0.0
    mlp_better_than_statsbomb = 0
    mlp_worse_than_statsbomb = 0
    mlp_similar_to_statsbomb = 0
    
    # Transformer comparison stats
    total_transformer_mse = 0.0
    total_transformer_relative_score = 0.0
    transformer_better_than_statsbomb = 0
    transformer_worse_than_statsbomb = 0
    transformer_similar_to_statsbomb = 0
    mlp_better_than_transformer = 0
    
    with torch.no_grad():
        for i, (match_id, shot_idx, goal_label) in enumerate(shot_samples):
            # Get MLP prediction (single shot event)
            shot_event = embeddings_dict[match_id][shot_idx:shot_idx + 1]
            X_mlp = torch.tensor(shot_event, dtype=torch.float32).unsqueeze(0).to(device)
            mlp_output = mlp_model(X_mlp)
            mlp_prediction = float(mlp_output.cpu().squeeze())
            
            # Get transformer prediction if available
            transformer_prediction = None
            if transformer_model:
                # Extract sequence ending with the shot event (sequence_length=200)
                sequence_length = 200
                sequence_start = max(0, shot_idx - sequence_length + 1)
                seq = transformer_embeddings_dict[match_id][sequence_start:shot_idx + 1]
                
                # Pad if sequence is shorter than required
                if len(seq) < sequence_length:
                    padding = np.zeros((sequence_length - len(seq), seq.shape[1]))
                    seq = np.concatenate([padding, seq], axis=0)
                
                X_transformer = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                transformer_output = transformer_model(X_transformer)
                transformer_prediction = float(transformer_output.cpu().squeeze())
            
            # Get original shot data (from unmasked events)
            shot_event_unmasked = events_dict[match_id][shot_idx]
            shot_data = decode_shot_event_data(shot_event_unmasked)
            
            # Track statistics
            if goal_label == 1:
                goal_count += 1
            else:
                no_goal_count += 1
            
            # Calculate MLP performance metrics
            statsbomb_xg = shot_data['statsbomb_xg']
            mlp_mse = (mlp_prediction - statsbomb_xg) ** 2
            total_mlp_mse += mlp_mse
            
            mlp_relative_score = calculate_relative_performance_score(mlp_prediction, statsbomb_xg, goal_label)
            total_mlp_relative_score += mlp_relative_score
            
            # Categorize MLP performance
            if mlp_relative_score > 0.1:
                mlp_better_than_statsbomb += 1
            elif mlp_relative_score < -0.1:
                mlp_worse_than_statsbomb += 1
            else:
                mlp_similar_to_statsbomb += 1
            
            # Calculate transformer performance metrics if available
            transformer_relative_score = None
            if transformer_prediction is not None:
                transformer_mse = (transformer_prediction - statsbomb_xg) ** 2
                total_transformer_mse += transformer_mse
                
                transformer_relative_score = calculate_relative_performance_score(transformer_prediction, statsbomb_xg, goal_label)
                total_transformer_relative_score += transformer_relative_score
                
                # Categorize transformer performance
                if transformer_relative_score > 0.1:
                    transformer_better_than_statsbomb += 1
                elif transformer_relative_score < -0.1:
                    transformer_worse_than_statsbomb += 1
                else:
                    transformer_similar_to_statsbomb += 1
                
                # Compare MLP vs Transformer
                if mlp_relative_score > transformer_relative_score:
                    mlp_better_than_transformer += 1
            
            # Display formatted shot information
            print(f"\nSAMPLE {i+1}/{len(shot_samples)}")
            print(format_shot_display_with_comparison(
                shot_data, mlp_prediction, transformer_prediction, goal_label, 
                mlp_relative_score, transformer_relative_score
            ))
    
    # Summary statistics
    num_samples_actual = len(shot_samples)
    avg_mlp_mse = total_mlp_mse / num_samples_actual
    mlp_rmse = np.sqrt(avg_mlp_mse)
    avg_mlp_relative_score = total_mlp_relative_score / num_samples_actual
    
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    print(f"Total Samples:     {num_samples_actual}")
    print(f"Goals:             {goal_count}")
    print(f"Non-Goals:         {no_goal_count}")
    print()
    print("📊 MLP BASELINE PERFORMANCE vs StatsBomb:")
    print(f"   Average Score:   {avg_mlp_relative_score:+.4f}")
    print(f"   Better:          {mlp_better_than_statsbomb}/{num_samples_actual} ({100*mlp_better_than_statsbomb/num_samples_actual:.1f}%) 🟢")
    print(f"   Similar:         {mlp_similar_to_statsbomb}/{num_samples_actual} ({100*mlp_similar_to_statsbomb/num_samples_actual:.1f}%) 🟡")
    print(f"   Worse:           {mlp_worse_than_statsbomb}/{num_samples_actual} ({100*mlp_worse_than_statsbomb/num_samples_actual:.1f}%) 🔴")
    print(f"   MSE vs StatsBomb: {avg_mlp_mse:.6f}")
    print(f"   RMSE vs StatsBomb: {mlp_rmse:.6f}")
    
    if transformer_model:
        avg_transformer_mse = total_transformer_mse / num_samples_actual
        transformer_rmse = np.sqrt(avg_transformer_mse)
        avg_transformer_relative_score = total_transformer_relative_score / num_samples_actual
        
        print()
        print("📊 TRANSFORMER PERFORMANCE vs StatsBomb:")
        print(f"   Average Score:   {avg_transformer_relative_score:+.4f}")
        print(f"   Better:          {transformer_better_than_statsbomb}/{num_samples_actual} ({100*transformer_better_than_statsbomb/num_samples_actual:.1f}%) 🟢")
        print(f"   Similar:         {transformer_similar_to_statsbomb}/{num_samples_actual} ({100*transformer_similar_to_statsbomb/num_samples_actual:.1f}%) 🟡")
        print(f"   Worse:           {transformer_worse_than_statsbomb}/{num_samples_actual} ({100*transformer_worse_than_statsbomb/num_samples_actual:.1f}%) 🔴")
        print(f"   MSE vs StatsBomb: {avg_transformer_mse:.6f}")
        print(f"   RMSE vs StatsBomb: {transformer_rmse:.6f}")
        
        print()
        print("⚡ MLP vs TRANSFORMER COMPARISON:")
        print(f"   MLP Better:      {mlp_better_than_transformer}/{num_samples_actual} ({100*mlp_better_than_transformer/num_samples_actual:.1f}%) 🟢")
        print(f"   Transformer Better: {num_samples_actual - mlp_better_than_transformer}/{num_samples_actual} ({100*(num_samples_actual - mlp_better_than_transformer)/num_samples_actual:.1f}%) 🔴")
        print(f"   Performance Gap:  {avg_transformer_relative_score - avg_mlp_relative_score:+.4f}")
        
    print("="*100)
    
    logger.info("Detailed MLP baseline evaluation completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed MLP Baseline Model Evaluation")
    parser.add_argument("--mlp-model-path", type=str, required=True, help="Path to trained MLP baseline model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing match CSV files")
    parser.add_argument("--encoder-path", type=str, required=True, help="Path to pretrained encoder")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of matches for training (to exclude from evaluation)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proportion of matches for validation (to exclude from evaluation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/test split")
    parser.add_argument("--transformer-model-path", type=str, help="Optional path to transformer model for comparison")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    
    # Run evaluation
    evaluate_mlp_baseline_detailed(
        mlp_model_path=args.mlp_model_path,
        data_dir=args.data_dir,
        encoder_path=args.encoder_path,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        transformer_model_path=args.transformer_model_path
    )