#!/usr/bin/env python3
"""
Update Labels - Updates form quality labels in existing data files

This script lets you update the form quality labels in an existing session's data files.
You can choose to:
1. Update all labels to a specific value (good, moderate, poor)
2. Set labels based on a pattern (e.g., alternating or threshold-based)
3. Apply custom labeling rules
"""

import os
import json
import pandas as pd
import argparse
import numpy as np
from datetime import datetime

def update_session_labels(session_id, label_type="pattern", value=None, data_dir="collected_data"):
    """
    Update form quality labels for a specific session
    
    Args:
        session_id: ID of the session to update
        label_type: Type of labeling to use:
                   - "fixed": Set all labels to a fixed value
                   - "pattern": Apply a repeating pattern based on rep_count
                   - "random": Randomly assign labels
                   - "threshold": Assign based on accelerometer thresholds
        value: For fixed type, the value to use (good, moderate, poor)
        data_dir: Directory containing data files
    """
    # Check if session exists
    meta_file = os.path.join(data_dir, f"session_{session_id}_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Session {session_id} not found in {data_dir}")
        return False
    
    # Load metadata
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
        exercise = metadata.get("exercise", "unknown")
    
    print(f"Updating labels for session {session_id} - {exercise}")
    
    # Check for required data files
    pose_file = os.path.join(data_dir, "pose_data", f"session_{session_id}_pose.json")
    mpu_file = os.path.join(data_dir, "mpu_data", f"session_{session_id}_mpu.csv")
    labeled_file = os.path.join(data_dir, "labeled_data", f"session_{session_id}_labeled.csv")
    
    has_pose = os.path.exists(pose_file)
    has_mpu = os.path.exists(mpu_file)
    has_labeled = os.path.exists(labeled_file)
    
    if not has_pose and not has_mpu:
        print("Error: No pose or MPU data found for this session")
        return False
    
    print(f"Found data files: pose={has_pose}, mpu={has_mpu}, labeled={has_labeled}")
    
    # Load data files
    pose_data = None
    mpu_data = None
    labeled_data = None
    
    if has_pose:
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        print(f"Loaded pose data: {len(pose_data)} entries")
    
    if has_mpu:
        mpu_data = pd.read_csv(mpu_file)
        print(f"Loaded MPU data: {len(mpu_data)} entries")
    
    if has_labeled:
        labeled_data = pd.read_csv(labeled_file)
        print(f"Loaded labeled data: {len(labeled_data)} entries")
    
    # Generate new labels according to the selected method
    new_labels = []
    
    if label_type == "fixed" and value:
        # Set all labels to the fixed value
        if value not in ["good", "moderate", "poor"]:
            print(f"Warning: Invalid label value '{value}'. Using 'moderate' instead.")
            value = "moderate"
        
        print(f"Setting all labels to '{value}'")
        new_labels = [value] * 10000  # More than enough for any dataset
    
    elif label_type == "pattern":
        # Create a pattern based on repetitions
        pattern = ["good", "moderate", "poor", "moderate"]
        new_labels = []
        
        # If we have MPU data, use rep count to determine labels
        if mpu_data is not None:
            max_rep = int(mpu_data["rep_count"].max())
            
            # Generate one label per rep
            for rep in range(max_rep + 1):
                label = pattern[rep % len(pattern)]
                count = len(mpu_data[mpu_data["rep_count"] == rep])
                new_labels.extend([label] * count)
        else:
            # Generate a repeating pattern based on data length
            data_len = len(pose_data) if pose_data is not None else 100
            for i in range(data_len):
                phase = (i % 20) / 20.0  # 20-point cycle
                if phase < 0.25:
                    new_labels.append("good")
                elif phase < 0.5:
                    new_labels.append("moderate")
                elif phase < 0.75:
                    new_labels.append("poor")
                else:
                    new_labels.append("moderate")
    
    elif label_type == "random":
        # Random labels with a distribution favoring "good"
        print("Applying random labeling (40% good, 40% moderate, 20% poor)")
        data_len = max(
            len(pose_data) if pose_data is not None else 0,
            len(mpu_data) if mpu_data is not None else 0,
            len(labeled_data) if labeled_data is not None else 0,
            100  # Default if no data
        )
        
        np.random.seed(42)  # For reproducibility
        for _ in range(data_len):
            rand_val = np.random.random()
            if rand_val < 0.4:
                new_labels.append("good")
            elif rand_val < 0.8:
                new_labels.append("moderate")
            else:
                new_labels.append("poor")
    
    elif label_type == "threshold":
        # Use accelerometer magnitude to determine quality
        print("Applying threshold-based labeling using accelerometer data")
        if mpu_data is None:
            print("Error: Need MPU data for threshold-based labeling")
            return False
        
        # Compute thresholds based on data
        if "magnitude" in mpu_data.columns:
            mag_values = mpu_data["magnitude"].values
            mag_mean = np.mean(mag_values)
            mag_std = np.std(mag_values)
            
            good_threshold = mag_mean + 0.5 * mag_std
            poor_threshold = mag_mean - 0.5 * mag_std
            
            for i in range(len(mpu_data)):
                mag = mpu_data.loc[i, "magnitude"]
                if mag > good_threshold:
                    new_labels.append("good")
                elif mag < poor_threshold:
                    new_labels.append("poor")
                else:
                    new_labels.append("moderate")
        else:
            # Use total acceleration if magnitude not available
            for i in range(len(mpu_data)):
                accel_x = mpu_data.loc[i, "accel_x"]
                accel_y = mpu_data.loc[i, "accel_y"]
                accel_z = mpu_data.loc[i, "accel_z"]
                
                # Compute magnitude
                mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                
                if mag > 1.5:
                    new_labels.append("good")
                elif mag < 1.0:
                    new_labels.append("poor")
                else:
                    new_labels.append("moderate")
    
    else:
        print(f"Error: Unknown label type '{label_type}'")
        return False
    
    # Update data files
    updated_files = []
    
    # Update pose data file
    if has_pose:
        for i in range(min(len(pose_data), len(new_labels))):
            pose_data[i]["form_quality"] = new_labels[i]
        
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        updated_files.append("pose_data")
    
    # Update MPU data file
    if has_mpu:
        for i in range(min(len(mpu_data), len(new_labels))):
            mpu_data.loc[i, "form_quality"] = new_labels[i]
        
        mpu_data.to_csv(mpu_file, index=False)
        updated_files.append("mpu_data")
    
    # Update or create labeled data file
    if has_labeled:
        # Update existing file
        for i in range(min(len(labeled_data), len(new_labels))):
            labeled_data.loc[i, "label"] = new_labels[i]
        
        labeled_data.to_csv(labeled_file, index=False)
        updated_files.append("updated_labeled_data")
    else:
        # Create new labeled file based on MPU data (if available)
        if has_mpu:
            labeled_rows = []
            for i in range(len(mpu_data)):
                if i < len(new_labels):
                    row = mpu_data.iloc[i].copy()
                    row["label"] = new_labels[i]
                    labeled_rows.append(row)
            
            labeled_df = pd.DataFrame(labeled_rows)
            labeled_df.to_csv(labeled_file, index=False)
            updated_files.append("created_labeled_data")
    
    print(f"Successfully updated labels in: {', '.join(updated_files)}")
    
    # Update metadata to reflect changes
    timestamp = datetime.now().isoformat()
    if "label_updates" not in metadata:
        metadata["label_updates"] = []
    
    metadata["label_updates"].append({
        "timestamp": timestamp,
        "method": label_type,
        "value": value,
        "updated_files": updated_files
    })
    
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Update form quality labels in existing data files")
    parser.add_argument("--session", type=str, required=True, 
                        help="Session ID to update (required)")
    parser.add_argument("--type", choices=["fixed", "pattern", "random", "threshold"], 
                        default="pattern", help="Labeling method to use")
    parser.add_argument("--value", type=str, help="Label value for fixed labeling (good, moderate, poor)")
    parser.add_argument("--data-dir", type=str, default="collected_data", 
                        help="Directory containing data files")
    
    args = parser.parse_args()
    
    if args.type == "fixed" and not args.value:
        print("Error: Must specify --value for fixed labeling")
        parser.print_help()
        return
    
    update_session_labels(args.session, args.type, args.value, args.data_dir)

if __name__ == "__main__":
    main() 