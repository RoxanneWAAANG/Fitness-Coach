#!/usr/bin/env python3
"""
Create Labeled Data - Utility to process existing data and generate labeled datasets

This script:
1. Finds all existing data collection sessions
2. Checks if they have labeled data files
3. Creates labeled data files for any sessions that don't have them
"""

import os
import json
import argparse
from data_collector import DataCollector

def process_all_sessions(data_dir="collected_data"):
    """Process all sessions in the data directory"""
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return
        
    # Get all metadata files
    meta_files = [f for f in os.listdir(data_dir) 
                  if f.startswith("session_") and f.endswith("_meta.json")]
    
    if not meta_files:
        print("No sessions found")
        return
        
    print(f"Found {len(meta_files)} sessions to process")
    
    # Create a data collector instance
    collector = DataCollector(data_dir=data_dir)
    
    # Process each session
    for meta_file in meta_files:
        session_id = meta_file.split("_")[1]
        labeled_file = os.path.join(data_dir, "labeled_data", f"session_{session_id}_labeled.csv")
        
        # Check if labeled file already exists
        if os.path.exists(labeled_file):
            print(f"Session {session_id} already has labeled data")
            continue
            
        # Check if pose and MPU data exist
        pose_file = os.path.join(data_dir, "pose_data", f"session_{session_id}_pose.json")
        mpu_file = os.path.join(data_dir, "mpu_data", f"session_{session_id}_mpu.csv")
        
        has_pose = os.path.exists(pose_file)
        has_mpu = os.path.exists(mpu_file)
        
        if not has_pose and not has_mpu:
            print(f"Session {session_id} has no pose or MPU data to process")
            continue
            
        # Read metadata
        try:
            with open(os.path.join(data_dir, meta_file), 'r') as f:
                metadata = json.load(f)
                exercise = metadata.get("exercise", "unknown")
                
            print(f"Processing session {session_id} for exercise: {exercise}")
            
            # Create the labeled dataset
            collector.current_session = session_id
            
            # Create the labeled data file
            success = collector.create_labeled_dataset(labeled_file)
            
            if success:
                print(f"Created labeled data file for session {session_id}")
            else:
                print(f"Failed to create labeled data for session {session_id}")
                
        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Create labeled data files for machine learning")
    parser.add_argument("--data-dir", type=str, default="collected_data", 
                        help="Directory containing collected data")
    parser.add_argument("--session", type=str, 
                        help="Process only a specific session ID")
    
    args = parser.parse_args()
    
    if args.session:
        # Process a specific session
        data_dir = args.data_dir
        session_id = args.session
        labeled_file = os.path.join(data_dir, "labeled_data", f"session_{session_id}_labeled.csv")
        
        collector = DataCollector(data_dir=data_dir)
        collector.current_session = session_id
        
        success = collector.create_labeled_dataset(labeled_file)
        if success:
            print(f"Created labeled data file for session {session_id}")
        else:
            print(f"Failed to create labeled data for session {session_id}")
    else:
        # Process all sessions
        process_all_sessions(args.data_dir)
    
if __name__ == "__main__":
    main() 