# Personalized Fitness Coach

A Raspberry Pi-based system that tracks and evaluates exercise form using computer vision and motion sensors, providing real-time feedback through an LCD display and LED indicators.

## Project Overview

This system combines pose detection (using MoveNet) with motion tracking (MPU6050 sensor) to:
- Collect multi-modal exercise data (visual pose + motion)
- Train machine learning models for form quality assessment
- Provide real-time feedback during workouts
- Support multiple exercise types (squats, pushups, bicep curls)

## Hardware Requirements

- Raspberry Pi 4
- Raspberry Pi Camera Module
- MPU6050 accelerometer/gyroscope sensor (I2C address: `0x53`)
- I2C LCD display (I2C address: `0x27`)
- 3 LEDs for feedback:
  - Blue LED (GPIO 16): Good form
  - Green LED (GPIO 20): Moderate form
  - Yellow LED (GPIO 21): Poor form
- Breadboard and jumper wires

## Software Dependencies

- Python==3.10
- tflite-runtime (for MoveNet model)
- OpenCV (for image processing)
- PyTorch (for neural network models)
- scikit-learn (for RandomForest models)
- pandas, numpy (for data processing)
- RPLCD (for LCD control)
- Picamera2 (for camera access)
- RPi.GPIO (for LED control)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RoxanneWAAANG/Fitness-Coach.git
   cd Fitness-Coach
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MoveNet TFLite model:
   ```bash
   mkdir -p models
   wget -O models/movenet.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
   ```

## Data Collection

Use the `multi_collector.py` module to gather training data:

```bash
python run_multi_collector.py
```

This will:
- Prompt you to select an exercise type
- Collect user information (height, weight, experience level)
- Record sensor data and camera frames
- Save labeled data for model training

The data is organized into:
- `collected_data/pose_data/`: JSON files with pose keypoints
- `collected_data/mpu_data/`: CSV files with accelerometer/gyroscope data
- `collected_data/image_data/`: Captured images
- `collected_data/labeled_data/`: Processed data ready for training

To view data, use `python utils/data_viewer.py`.

## Model Training

### How to Run
Train form quality assessment models using:

```bash
python model_trainer.py --exercise squat --model random_forest
```

Options:
- `--exercise`: Exercise type (squat, pushup, bicep_curl)
- `--model`: Model type (random_forest, pytorch)
- `--list-exercises`: List available exercises in the collected data
- `--data-dir`: Directory for input data (default: collected_data)
- `--models-dir`: Directory for saving models (default: models)

Models will be saved to the `models/` directory.

### Model Results

Below are the accuracy results from training different models:

| Exercise Type | Model Type | Accuracy | Notes |
|---------------|------------|----------|-------|
| Bicep Curl | PyTorch Neural Network | 100.00% | Loss: 0.0261 after 50 epochs |
| Bicep Curl | Random Forest | 100.00% | Perfect precision/recall for all classes |
| Pushup | PyTorch Neural Network | 99.13% | Loss: 0.0240 after 50 epochs |
| Pushup | Random Forest | 100.00% | Perfect precision/recall for all classes |
| Squat | PyTorch Neural Network | 43.48% | Loss: 0.9960 after 50 epochs |
| Squat | Random Forest | 24.35% | Low precision/recall across all classes |

> **Note on Squat Models**: The lower accuracy for squat models suggests that more high-quality training data may be needed, or that the current feature set doesn't adequately capture the complexity of squat form variations. Consider collecting more diverse data or adding additional sensors for better results.

## Running the Fitness Coach

Start the interactive fitness coach:

```bash
python fitness_coach_cli.py
```

This will:
1. Present a menu to select exercise mode or auto-detection
2. Monitor your movements in real-time
3. Provide feedback through:
   - LCD display showing exercise type and form quality
   - LED indicators (Blue = Good, Green = Moderate, Yellow = Poor)
   - Terminal output

Press Ctrl+C to return to the menu or exit the application.

## Project Structure

- `data_collector.py`: Core class for data collection using camera and MPU6050
- `run_data_collector.py`: Interactive script for collecting exercise data
- `model_trainer.py`: Trains and evaluates form quality models
- `fitness_coach_cli.py`: Main application with real-time feedback
- `models/`: Directory for storing trained models and the MoveNet model
- `collected_data/`: Directory for storing collected exercise data

## Future Work

The current system provides a prototype for exercise analysis, but several enhancements could further improve its capabilities:

1. Improved Models Performance:

    - Collect more diverse data across different body types and skill levels
    - Experiment with different feature engineering approaches
    - Try more complex models like CNNs or transformers for time-series data


2. Extended Exercise Library:

    - Add support for additional exercises (deadlifts, lunges, shoulder press, etc.)
    - Create compound exercise workflows for circuit training


3. Hardware Improvements:

    - Add multiple sensors for better movement tracking
    - Use camera holder for better detection

## Troubleshooting

- **Hardware error**:
   - Test with `python utils/test_hardware.py`

- **Model training errors**:
  - Ensure sufficient data has been collected
  - Check log files in `training_logs/` directory

## License

MIT License – see the LICENSE file for full details.

