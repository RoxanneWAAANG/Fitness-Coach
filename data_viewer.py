#!/usr/bin/env python3
"""
Data Viewer - Tool for visualizing collected fitness data

This script allows you to:
1. View session summaries
2. Visualize pose data and motion patterns
3. Analyze form quality distribution
4. Prepare data for model training
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

class DataViewer:
    def __init__(self, data_dir="collected_data"):
        self.data_dir = data_dir
        self.sessions = self.load_sessions()
        
    def load_sessions(self):
        """Load all session metadata"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return []
            
        # Get all metadata files
        meta_files = [f for f in os.listdir(self.data_dir) 
                      if f.startswith("session_") and f.endswith("_meta.json")]
        
        sessions = []
        for meta_file in meta_files:
            try:
                with open(os.path.join(self.data_dir, meta_file), 'r') as f:
                    metadata = json.load(f)
                    session_id = meta_file.split("_")[1]
                    
                    # Check if data files exist
                    has_pose = os.path.exists(os.path.join(self.data_dir, "pose_data", f"session_{session_id}_pose.json"))
                    has_mpu = os.path.exists(os.path.join(self.data_dir, "mpu_data", f"session_{session_id}_mpu.csv"))
                    has_labeled = os.path.exists(os.path.join(self.data_dir, "labeled_data", f"session_{session_id}_labeled.csv"))
                    
                    session_info = {
                        "id": session_id,
                        "meta": metadata,
                        "has_pose": has_pose,
                        "has_mpu": has_mpu,
                        "has_labeled": has_labeled
                    }
                    sessions.append(session_info)
            except Exception as e:
                print(f"Error loading metadata file {meta_file}: {e}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x["meta"].get("start_time", ""), reverse=True)
        return sessions
    
    def list_sessions(self):
        """Display a summary of all sessions"""
        if not self.sessions:
            print("No sessions found")
            return
            
        print(f"\n{'='*80}")
        print(f"{'ID':<15} {'Exercise':<15} {'Date':<20} {'Data Points':<15} {'Available Data'}")
        print(f"{'-'*80}")
        
        for session in self.sessions:
            meta = session["meta"]
            session_id = session["id"]
            date_str = "Unknown"
            if "start_time" in meta:
                date_str = datetime.fromisoformat(meta["start_time"]).strftime("%Y-%m-%d %H:%M")
            
            data_points = meta.get("data_points", "?")
            data_types = []
            if session["has_pose"]:
                data_types.append("pose")
            if session["has_mpu"]:
                data_types.append("mpu")
            if session["has_labeled"]:
                data_types.append("label")
                
            print(f"{session_id:<15} {meta.get('exercise', 'Unknown'):<15} {date_str:<20} {data_points:<15} {', '.join(data_types)}")
        
        print(f"{'='*80}")
    
    def load_session_data(self, session_id):
        """Load all data for a specific session"""
        # Find the session
        session = next((s for s in self.sessions if s["id"] == session_id), None)
        if not session:
            print(f"Session {session_id} not found")
            return None
            
        data = {"meta": session["meta"], "id": session_id}
        
        # Load pose data
        if session["has_pose"]:
            pose_file = os.path.join(self.data_dir, "pose_data", f"session_{session_id}_pose.json")
            try:
                with open(pose_file, 'r') as f:
                    data["pose"] = json.load(f)
            except Exception as e:
                print(f"Error loading pose data: {e}")
        
        # Load MPU data
        if session["has_mpu"]:
            mpu_file = os.path.join(self.data_dir, "mpu_data", f"session_{session_id}_mpu.csv")
            try:
                data["mpu"] = pd.read_csv(mpu_file)
            except Exception as e:
                print(f"Error loading MPU data: {e}")
        
        # Load labeled data
        if session["has_labeled"]:
            labeled_file = os.path.join(self.data_dir, "labeled_data", f"session_{session_id}_labeled.csv")
            try:
                data["label"] = pd.read_csv(labeled_file)
            except Exception as e:
                print(f"Error loading labeled data: {e}")
        
        return data
    
    def visualize_session(self, session_id):
        """Visualize data from a specific session"""
        data = self.load_session_data(session_id)
        if not data:
            return
            
        print(f"\n{'='*80}")
        print(f"Session {session_id} - {data['meta'].get('exercise', 'Unknown')}")
        print(f"Started: {data['meta'].get('start_time', 'Unknown')}")
        print(f"{'='*80}")
        
        # Display user info if available
        if "user_info" in data["meta"] and data["meta"]["user_info"]:
            print("User Info:")
            for key, value in data["meta"]["user_info"].items():
                print(f"  {key}: {value}")
            print()
        
        # Display session notes if available
        if "notes" in data["meta"]:
            print("Session Notes:")
            for key, value in data["meta"]["notes"].items():
                print(f"  {key}: {value}")
            print()
        
        # Plot MPU data if available
        if "mpu" in data and not data["mpu"].empty:
            self.plot_mpu_data(data["mpu"])
        
        # Plot form quality distribution if available
        if "pose" in data and data["pose"]:
            self.analyze_form_quality(data["pose"])
        
        # Show summary statistics
        if "mpu" in data and not data["mpu"].empty:
            self.show_statistics(data)
    
    def plot_mpu_data(self, mpu_data):
        """Plot accelerometer and gyroscope data"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot accelerometer data
            plt.subplot(2, 1, 1)
            plt.plot(mpu_data["timestamp"], mpu_data["accel_x"], label="X")
            plt.plot(mpu_data["timestamp"], mpu_data["accel_y"], label="Y")
            plt.plot(mpu_data["timestamp"], mpu_data["accel_z"], label="Z")
            if "magnitude" in mpu_data.columns:
                plt.plot(mpu_data["timestamp"], mpu_data["magnitude"], label="Magnitude", linewidth=2, color='black')
            plt.title("Accelerometer Data")
            plt.xlabel("Time")
            plt.ylabel("Acceleration")
            plt.legend()
            plt.grid(True)
            
            # Plot gyroscope data
            plt.subplot(2, 1, 2)
            plt.plot(mpu_data["timestamp"], mpu_data["gyro_x"], label="X")
            plt.plot(mpu_data["timestamp"], mpu_data["gyro_y"], label="Y")
            plt.plot(mpu_data["timestamp"], mpu_data["gyro_z"], label="Z")
            plt.title("Gyroscope Data")
            plt.xlabel("Time")
            plt.ylabel("Angular Velocity")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting MPU data: {e}")
    
    def analyze_form_quality(self, pose_data):
        """Analyze and visualize form quality distribution"""
        try:
            # Count form quality labels
            quality_counts = {}
            for datapoint in pose_data:
                quality = datapoint.get("form_quality", "unknown")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Plot distribution
            plt.figure(figsize=(8, 6))
            labels = list(quality_counts.keys())
            counts = list(quality_counts.values())
            
            # Sort by form quality (good, moderate, poor)
            quality_order = {"good": 0, "moderate": 1, "poor": 2, "unknown": 3}
            sorted_indices = sorted(range(len(labels)), key=lambda i: quality_order.get(labels[i], 99))
            sorted_labels = [labels[i] for i in sorted_indices]
            sorted_counts = [counts[i] for i in sorted_indices]
            
            colors = {
                "good": "green", 
                "moderate": "orange", 
                "poor": "red", 
                "unknown": "gray"
            }
            bar_colors = [colors.get(label, "blue") for label in sorted_labels]
            
            plt.bar(sorted_labels, sorted_counts, color=bar_colors)
            plt.title("Form Quality Distribution")
            plt.xlabel("Form Quality")
            plt.ylabel("Count")
            
            # Add percentage labels
            total = sum(counts)
            for i, count in enumerate(sorted_counts):
                percentage = round(count / total * 100, 1)
                plt.text(i, count, f"{percentage}%", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error analyzing form quality: {e}")
    
    def show_statistics(self, data):
        """Show session statistics"""
        print("\nSession Statistics:")
        
        # Exercise duration
        if "meta" in data and "start_time" in data["meta"] and "end_time" in data["meta"]:
            start = datetime.fromisoformat(data["meta"]["start_time"])
            end = datetime.fromisoformat(data["meta"]["end_time"])
            duration = (end - start).total_seconds()
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Rep count
        if "mpu" in data and not data["mpu"].empty:
            max_rep = data["mpu"]["rep_count"].max()
            print(f"Maximum rep count: {max_rep}")
        
        # Data points
        if "meta" in data and "data_points" in data["meta"]:
            print(f"Total data points: {data['meta']['data_points']}")
        
        print()
    
    def export_for_training(self, session_ids=None):
        """Export data from specified sessions for model training"""
        if not session_ids:
            # Export all sessions
            session_ids = [s["id"] for s in self.sessions]
        
        combined_labeled_data = []
        
        for session_id in session_ids:
            data = self.load_session_data(session_id)
            if not data or "label" not in data:
                print(f"No labeled data found for session {session_id}")
                continue
                
            exercise_type = data["meta"].get("exercise", "unknown")
            labeled_data = data["label"]
            labeled_data["exercise"] = exercise_type
            labeled_data["session_id"] = session_id
            
            combined_labeled_data.append(labeled_data)
        
        if not combined_labeled_data:
            print("No data to export")
            return
            
        # Combine all data
        combined_df = pd.concat(combined_labeled_data, ignore_index=True)
        
        # Export to CSV
        export_dir = "training_data"
        os.makedirs(export_dir, exist_ok=True)
        export_file = os.path.join(export_dir, f"combined_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_df.to_csv(export_file, index=False)
        
        print(f"Exported {len(combined_df)} data points to {export_file}")

def main():
    parser = argparse.ArgumentParser(description="View and analyze collected fitness data")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    parser.add_argument("--view", type=str, help="View data for a specific session ID")
    parser.add_argument("--export", action="store_true", help="Export data for model training")
    parser.add_argument("--sessions", type=str, help="Comma-separated list of session IDs to export")
    parser.add_argument("--data-dir", type=str, default="collected_data", help="Data directory")
    
    args = parser.parse_args()
    
    viewer = DataViewer(data_dir=args.data_dir)
    
    if args.list:
        viewer.list_sessions()
    elif args.view:
        viewer.visualize_session(args.view)
    elif args.export:
        if args.sessions:
            session_ids = args.sessions.split(",")
            viewer.export_for_training(session_ids)
        else:
            viewer.export_for_training()
    else:
        # Interactive mode if no arguments provided
        while True:
            print("\nFitness Data Viewer")
            print("1. List all sessions")
            print("2. View session data")
            print("3. Export data for training")
            print("4. Exit")
            
            choice = input("Select an option (1-4): ")
            
            if choice == "1":
                viewer.list_sessions()
            elif choice == "2":
                if not viewer.sessions:
                    print("No sessions available")
                    continue
                    
                viewer.list_sessions()
                session_id = input("Enter session ID to view: ")
                viewer.visualize_session(session_id)
            elif choice == "3":
                if not viewer.sessions:
                    print("No sessions available")
                    continue
                    
                viewer.list_sessions()
                session_ids = input("Enter session IDs to export (comma-separated), or leave blank for all: ")
                if session_ids:
                    viewer.export_for_training(session_ids.split(","))
                else:
                    viewer.export_for_training()
            elif choice == "4":
                break
            else:
                print("Invalid choice")

if __name__ == "__main__":
    # Check if matplotlib is available, if not print helpful message
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for visualization.")
        print("Please install it using: pip install matplotlib")
        exit(1)
        
    main() 