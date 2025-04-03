# run_data_collection.py

import time
from collector import DataCollector  # Ensure this matches your filename

def main():
    collector = DataCollector()

    exercise = input("Enter exercise name (e.g., squat, pushup): ")
    height = input("Enter your height in cm: ")
    weight = input("Enter your weight in kg: ")

    try:
        height = float(height)
        weight = float(weight)
    except ValueError:
        print("Invalid height or weight input. Please enter numeric values.")
        return

    user_info = {"height": height, "weight": weight}
    session_id = collector.start_session(exercise_name=exercise, user_info=user_info)
    print(f"Started session {session_id}")

    try:
        while True:
            print("\nPress Ctrl+C to end session.")
            form_quality = input("Enter form quality (e.g., good, bad, average): ")
            rep_count = input("Enter rep count: ")

            success = collector.record_data_point(form_quality=form_quality, rep_count=rep_count)
            print("Data point recorded." if success else "Failed to record data point.")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSession ending...")

    collector.end_session()
    print("Session data saved.")

if __name__ == "__main__":
    main()
