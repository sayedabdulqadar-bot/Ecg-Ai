"""
ECG Data Plotter - Collects data from ESP32 AD8232 sensor
Reads ECG data from COM5, applies denoising, and generates plot
Output: new_image.png in ecg_reports folder
"""

import numpy as np
import matplotlib.pyplot as plt
import serial
import re
import os
import shutil
from datetime import datetime
from scipy.signal import butter, filtfilt

# --- Configuration ---
SERIAL_PORT = 'COM5'  # ESP32 serial port
BAUD_RATE = 115200
SAMPLE_RATE = 200  # Hz
PLOT_DURATION_SECONDS = 5
NUM_SAMPLES_PER_PLOT = SAMPLE_RATE * PLOT_DURATION_SECONDS
OUTPUT_FOLDER = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports"
STANDARDIZED_IMAGE_NAME = "new_image.png"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Storage for ECG samples
ecg_data = []


def denoise_ecg(signal, sample_rate=200, lowcut=0.5, highcut=40):
    """Apply bandpass filter to remove noise."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def generate_and_save_plot(data, filename=None):
    """Generate and save ECG plot."""
    if len(data) < NUM_SAMPLES_PER_PLOT:
        print(f"Not enough data: {len(data)} samples")
        return None
    
    # Take exactly NUM_SAMPLES_PER_PLOT samples
    data_segment = np.array(data[:NUM_SAMPLES_PER_PLOT])
    
    # Denoise the signal
    denoised = denoise_ecg(data_segment, sample_rate=SAMPLE_RATE)
    
    # Time axis
    time_axis = np.linspace(0, PLOT_DURATION_SECONDS, NUM_SAMPLES_PER_PLOT)
    
    # Create plot
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, denoised, color='red', linewidth=0.8)
    plt.title("ECG Signal (Denoised)", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"ecg_report_{timestamp}.png"
    timestamped_path = os.path.join(OUTPUT_FOLDER, timestamped_name)
    
    # Save timestamped version
    plt.savefig(timestamped_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save standardized filename for processing
    standardized_path = os.path.join(OUTPUT_FOLDER, STANDARDIZED_IMAGE_NAME)
    shutil.copyfile(timestamped_path, standardized_path)
    
    print(f"✓ Saved ECG report to {timestamped_path}")
    print(f"✓ Copied to {standardized_path}")
    
    return standardized_path


def collect_ecg_data():
    """Main function to collect ECG data and generate plot."""
    # --- Serial Port Setup ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✓ Connected to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud.")
        print(f"✓ Collecting {PLOT_DURATION_SECONDS}s segments at {SAMPLE_RATE}Hz")
        print(f"✓ Output folder: {OUTPUT_FOLDER}/")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Warning: Could not open serial port {SERIAL_PORT}. {e}")
        ser = None

    # --- Main Loop ---
    print("Collecting ECG data from ESP32... Press Ctrl+C to stop.\n")
    
    try:
        sample_count = 0
        
        while True:
            if ser:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                match = re.match(r'>ECG:(\d+)', line)
                
                if match:
                    try:
                        ecg_value = int(match.group(1))
                        ecg_data.append(ecg_value)
                        sample_count += 1
                    except ValueError:
                        pass
            else:
                # Simulate ECG data for testing (if serial unavailable)
                t = np.linspace(0, PLOT_DURATION_SECONDS, NUM_SAMPLES_PER_PLOT)
                simulated = 2048 + 1000 * np.sin(2 * np.pi * 1.2 * t) + 150 * np.random.randn(len(t))
                ecg_data.extend([int(v) for v in simulated])
                print("Serial not available – generated simulated ECG for testing.")
            
            # Check if enough data collected
            if len(ecg_data) >= NUM_SAMPLES_PER_PLOT:
                print("\n" + "=" * 60)
                generated_filepath = generate_and_save_plot(ecg_data)
                print("=" * 60)
                print(f"✓ ECG image ready: {generated_filepath}")
                print("=" * 60 + "\n")
                break

    except KeyboardInterrupt:
        print("\n\n✓ Stopping data collection.")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser and ser.is_open:
            ser.close()
            print("✓ Serial port closed.")
    
    print("\nECG data collection complete!")
    print(f"Output saved in: {OUTPUT_FOLDER}/")
    
    return os.path.join(OUTPUT_FOLDER, STANDARDIZED_IMAGE_NAME)


if __name__ == "__main__":
    output_path = collect_ecg_data()
    print(f"\nReady for processing: {output_path}")