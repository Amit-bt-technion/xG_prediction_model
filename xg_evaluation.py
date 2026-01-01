"""
xg_evaluation.py
---------------
Specialized evaluation functionality for xG prediction models.
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
from sequence_classifier.sequence_transformer import ContinuousValueTransformer


def decode_shot_event_data(event_vector: np.ndarray) -> Dict:
    """
    Decode shot event data into human-readable format.
    
    Args:
        event_vector: The raw event vector from events_dict
        
    Returns:
        Dictionary containing decoded shot information
    """
    # Based on tokenized_eventpy.py structure
    # Common features (indices 0-14):
    common_data = {
        'type_id': event_vector[0],  # Should be ~0.227 (10/44) for shot events
        'play_pattern_id': event_vector[1],
        'location_x': event_vector[2] * 120,  # Denormalize from 0-1 to 0-120
        'location_y': event_vector[3] * 80,   # Denormalize from 0-1 to 0-80
        'duration': event_vector[4] * 3,      # Denormalize from 0-1 to 0-3 seconds
        'under_pressure': bool(event_vector[5] > 0.5),
        'out': bool(event_vector[6] > 0.5),
        'counterpress': bool(event_vector[7] > 0.5),
        'period': int(event_vector[8] * 5) + 1,  # Approximate denormalization
        'second': int(event_vector[9] * 60),     # Denormalize to 0-60 seconds
        'position_id': int(event_vector[10] * 25) + 1,  # Approximate denormalization
        'minute': event_vector[11] * 60,  # Special parser for minute
        'team_id': int(event_vector[12]),  # 0 or 1
        'possession_team_id': int(event_vector[13]),  # 0 or 1
        'player_designated_position': event_vector[14],
    }
    
    # Shot-specific features (starting at index 71):
    shot_data = {
        'shot_type_id': event_vector[71],
        'end_location_x': event_vector[72] * 120,  # Denormalize
        'end_location_y': event_vector[73] * 80,   # Denormalize  
        'end_location_z': event_vector[74] * 5,    # Denormalize to 0-5 meters
        'aerial_won': bool(event_vector[75] > 0.5),
        'follows_dribble': bool(event_vector[76] > 0.5),
        'first_time': bool(event_vector[77] > 0.5),
        'open_goal': bool(event_vector[78] > 0.5),
        'statsbomb_xg': event_vector[79],  # This is the original StatsBomb xG
        'deflected': bool(event_vector[80] > 0.5),
        'technique_id': event_vector[81],
        'body_part_id': event_vector[82],
        'outcome_id': event_vector[83],  # 0.25 indicates goal (outcome 97)
    }
    
    return {**common_data, **shot_data}


def calculate_relative_performance_score(model_xg: float, statsbomb_xg: float, actual_outcome: int) -> float:
    """
    Calculate a relative performance score comparing model vs StatsBomb xG.
    
    The score measures how much better (or worse) the model prediction is compared to 
    StatsBomb xG relative to the actual outcome.
    
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
    # (StatsBomb_Error - Model_Error) / max(errors, epsilon)
    # Positive means model is better, negative means model is worse
    denominator = max(statsbomb_error, model_error, epsilon)
    score = (statsbomb_error - model_error) / denominator
    
    return score


def format_shot_display(shot_data: Dict, model_prediction: float, goal_label: int, relative_performance_score: float) -> str:
    """
    Format shot data for comprehensive display.
    
    Args:
        shot_data: Dictionary of decoded shot information
        model_prediction: Raw model prediction (0-1)
        goal_label: Actual goal outcome (0 or 1)
        relative_performance_score: Score comparing model vs StatsBomb performance
        
    Returns:
        Formatted string for display
    """
    outcome_text = "GOAL" if goal_label == 1 else "NO GOAL"
    
    display = f"""
{'='*80}
SHOT OUTCOME: {outcome_text}
{'='*80}

🎯 PREDICTIONS:
   Model xG:     {model_prediction:.4f}
   StatsBomb xG: {shot_data['statsbomb_xg']:.4f}
   Difference:   {abs(model_prediction - shot_data['statsbomb_xg']):.4f}
   
📊 PERFORMANCE COMPARISON:
   Relative Score: {relative_performance_score:+.3f} {'🟢' if relative_performance_score > 0.1 else '🔴' if relative_performance_score < -0.1 else '🟡'}
   Interpretation: {'Model BETTER than StatsBomb' if relative_performance_score > 0.1 else 'Model WORSE than StatsBomb' if relative_performance_score < -0.1 else 'Model SIMILAR to StatsBomb'}

📍 LOCATION & TIMING:
   Shot Location:  ({shot_data['location_x']:.1f}, {shot_data['location_y']:.1f})
   End Location:   ({shot_data['end_location_x']:.1f}, {shot_data['end_location_y']:.1f}, {shot_data['end_location_z']:.1f}m)
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


def sample_balanced_shots(
    events_dict: Dict[str, np.ndarray],
    target_goals: int = 10,
    target_no_goals: int = 10,
    sequence_length: int = 200
) -> List[Tuple[str, int, int]]:
    """
    Sample a balanced set of shots (goals and non-goals).
    
    Args:
        events_dict: Dictionary of match events
        target_goals: Number of goal shots to sample
        target_no_goals: Number of non-goal shots to sample
        sequence_length: Required sequence length before shot
        
    Returns:
        List of (match_id, shot_index, goal_label) tuples
    """
    goal_samples = []
    no_goal_samples = []
    
    # Collect all valid shots
    for match_id, match_events in events_dict.items():
        num_events = len(match_events)
        
        # Look for shots with sufficient preceding events
        for i in range(sequence_length - 1, num_events):
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

    min_num_of_samples = min(len(goal_samples), len(no_goal_samples))

    # Take the required number of each
    selected_goals = goal_samples[:min_num_of_samples]
    selected_no_goals = no_goal_samples[:min_num_of_samples]
    
    # Combine and shuffle the final list
    all_samples = selected_goals + selected_no_goals
    random.shuffle(all_samples)
    
    return all_samples


def evaluate_xg_model_detailed(
    model_path: str,
    data_dir: str,
    encoder_path: str,
    cache_dir: str = "cache",
    sequence_length: int = 200,
    device: Optional[torch.device] = None,
    num_samples: int = 20,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Detailed evaluation of xG model with sample predictions and StatsBomb comparison.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing match CSV files
        encoder_path: Path to pretrained encoder
        cache_dir: Cache directory for embeddings
        sequence_length: Sequence length used in training
        device: Device to run evaluation on
        num_samples: Total number of samples to evaluate (should be even)
        train_ratio: Proportion of matches used for training (to exclude from evaluation)
        val_ratio: Proportion of matches used for validation (to exclude from evaluation)
        seed: Random seed for reproducible train/test split
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
    
    logger.info(f"Starting detailed xG evaluation with {target_goals} goals and {target_no_goals} non-goals")
    
    # Load data with masking for xG prediction
    mask_list = [4, 6, 7, 72, 73, 74, 75, 78, 79, 80, 83]  # out, end_location[0,1,2], aerial_won, open_goal, statsbomb_xg, deflected, outcome.id
    
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
        task="xg_prediction"
    )
    
    # Apply same train/test split as training to get test set matches only
    logger.info("Applying train/test split to ensure evaluation uses only test set...")
    match_ids = list(events_dict.keys())
    
    # Set random seed for reproducible split (same as training)
    random_state = random.getstate()
    random.seed(seed)
    random.shuffle(match_ids)
    random.setstate(random_state)  # Restore original random state
    
    # Split match IDs into train, validation, and test sets (same logic as create_data_loaders)
    num_matches = len(match_ids)
    train_size = int(train_ratio * num_matches)
    val_size = int(val_ratio * num_matches)
    
    test_match_ids = match_ids[train_size + val_size:]
    logger.info(f"Using {len(test_match_ids)} test matches out of {num_matches} total matches")
    logger.info(f"Excluded {train_size} training matches and {val_size} validation matches")
    
    # Filter events_dict to only include test matches
    test_events_dict = {match_id: events_dict[match_id] for match_id in test_match_ids}
    
    # Sample balanced shots from test set only
    logger.info("Sampling balanced shots from TEST SET ONLY...")
    shot_samples = sample_balanced_shots(
        events_dict=test_events_dict,
        target_goals=target_goals,
        target_no_goals=target_no_goals,
        sequence_length=sequence_length
    )
    
    if len(shot_samples) < num_samples:
        logger.warning(f"Only found {len(shot_samples)} valid shots, expected {num_samples}")
    
    # Validation: Ensure all shot samples are from test set
    test_match_ids_set = set(test_match_ids)
    for match_id, _, _ in shot_samples:
        if match_id not in test_match_ids_set:
            raise ValueError(f"CRITICAL ERROR: Found shot sample from non-test match {match_id}. This violates the test set isolation requirement!")
    
    logger.info(f"✓ VALIDATION PASSED: All {len(shot_samples)} shot samples are from test set matches only")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ContinuousValueTransformer(
        embedding_dim=32,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Evaluate each sample
    logger.info("Evaluating samples...")
    print("\n" + "="*100)
    print(f"XG MODEL EVALUATION - {len(shot_samples)} SAMPLES")
    print("="*100)
    
    goal_count = 0
    no_goal_count = 0
    total_model_mse = 0.0
    total_relative_score = 0.0
    better_than_statsbomb = 0
    worse_than_statsbomb = 0
    similar_to_statsbomb = 0
    
    with torch.no_grad():
        for i, (match_id, shot_idx, goal_label) in enumerate(shot_samples):
            # Extract sequence for model prediction (using masked embeddings)
            sequence_start = shot_idx - sequence_length + 1
            seq = embeddings_dict[match_id][sequence_start:shot_idx + 1]
            
            # Get model prediction
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            model_output = model(X)
            model_prediction = float(model_output.cpu().squeeze())
            
            # Get original shot data (from unmasked events)
            shot_event = events_dict[match_id][shot_idx]
            shot_data = decode_shot_event_data(shot_event)
            
            # Track statistics
            if goal_label == 1:
                goal_count += 1
            else:
                no_goal_count += 1
            
            # Calculate MSE against StatsBomb xG
            statsbomb_xg = shot_data['statsbomb_xg']
            mse = (model_prediction - statsbomb_xg) ** 2
            total_model_mse += mse
            
            # Calculate relative performance score
            relative_score = calculate_relative_performance_score(model_prediction, statsbomb_xg, goal_label)
            total_relative_score += relative_score
            
            # Categorize performance
            if relative_score > 0.1:
                better_than_statsbomb += 1
            elif relative_score < -0.1:
                worse_than_statsbomb += 1
            else:
                similar_to_statsbomb += 1
            
            # Display formatted shot information
            print(f"\nSAMPLE {i+1}/{len(shot_samples)}")
            print(format_shot_display(shot_data, model_prediction, goal_label, relative_score))
    
    # Summary statistics
    avg_mse = total_model_mse / len(shot_samples)
    rmse = np.sqrt(avg_mse)
    avg_relative_score = total_relative_score / len(shot_samples)
    
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    print(f"Total Samples:     {len(shot_samples)}")
    print(f"Goals:             {goal_count}")
    print(f"Non-Goals:         {no_goal_count}")
    print()
    print("📊 RELATIVE PERFORMANCE vs StatsBomb:")
    print(f"   Average Score:   {avg_relative_score:+.4f}")
    print(f"   Better:          {better_than_statsbomb}/{len(shot_samples)} ({100*better_than_statsbomb/len(shot_samples):.1f}%) 🟢")
    print(f"   Similar:         {similar_to_statsbomb}/{len(shot_samples)} ({100*similar_to_statsbomb/len(shot_samples):.1f}%) 🟡")
    print(f"   Worse:           {worse_than_statsbomb}/{len(shot_samples)} ({100*worse_than_statsbomb/len(shot_samples):.1f}%) 🔴")
    print()
    print("📈 TRADITIONAL METRICS:")
    print(f"   Model vs StatsBomb MSE:  {avg_mse:.6f}")
    print(f"   Model vs StatsBomb RMSE: {rmse:.6f}")
    print("="*100)
    
    logger.info("Detailed xG evaluation completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed xG Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing match CSV files")
    parser.add_argument("--encoder-path", type=str, required=True, help="Path to pretrained encoder")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--sequence-length", type=int, default=200, help="Sequence length")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of matches for training (to exclude from evaluation)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proportion of matches for validation (to exclude from evaluation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/test split")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    
    # Run evaluation
    evaluate_xg_model_detailed(
        model_path=args.model_path,
        data_dir=args.data_dir,
        encoder_path=args.encoder_path,
        cache_dir=args.cache_dir,
        sequence_length=args.sequence_length,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    ) 