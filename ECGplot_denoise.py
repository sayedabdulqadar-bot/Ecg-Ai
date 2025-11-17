import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt

# --- Configuration ---
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART'  # Change this to your ESP32's COM port!
BAUD_RATE = 115200
SAMPLE_RATE = 500      # Hz (Must match ESP32's SAMPLE_RATE)
PLOT_DURATION_SECONDS = 5 # How many seconds of ECG data to plot per image
OUTPUT_FOLDER = 'ecg_reports'

# Calculate number of samples needed for one plot
NUM_SAMPLES_PER_PLOT = int(SAMPLE_RATE * PLOT_DURATION_SECONDS)

# Initialize data buffer
ecg_data = []

# Ensure output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- Signal Processing Functions ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    """
    Design a Butterworth lowpass filter
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def notch_filter(data, fs, freq=50.0, quality=30.0):
    """
    Apply notch filter to remove powerline interference (50/60 Hz)
    """
    nyquist = 0.5 * fs
    freq = freq / nyquist
    b, a = signal.iirnotch(freq, quality)
    return filtfilt(b, a, data)

def denoise_ecg_signal(raw_signal, fs=500):
    """
    Comprehensive ECG denoising pipeline
    Based on methods from the Nature article
    
    Steps:
    1. Remove baseline wander (high-pass filter)
    2. Remove powerline interference (notch filter)
    3. Remove high-frequency noise (low-pass filter)
    4. Optional: Median filter for impulse noise
    """
    
    # Convert to numpy array and normalize
    signal_array = np.array(raw_signal, dtype=float)
    
    # Step 1: Remove baseline wander with high-pass filter (0.5 Hz cutoff)
    b_high, a_high = butter(4, 0.5, btype='highpass', fs=fs)
    signal_baseline_removed = filtfilt(b_high, a_high, signal_array)
    
    # Step 2: Bandpass filter (0.5-40 Hz) - typical ECG frequency range
    # This removes both baseline wander and high-frequency noise
    b_band, a_band = butter_bandpass(0.5, 40, fs, order=4)
    signal_bandpass = filtfilt(b_band, a_band, signal_array)
    
    # Step 3: Remove 50/60 Hz powerline interference (notch filter)
    # Try both 50Hz and 60Hz
    signal_notch = notch_filter(signal_bandpass, fs, freq=60.0, quality=30)
    signal_notch = notch_filter(signal_notch, fs, freq=50.0, quality=30)
    
    # Step 4: Optional median filter to remove impulse noise (kernel size must be odd)
    # Use small kernel to preserve QRS complex
    signal_denoised = medfilt(signal_notch, kernel_size=3)
    
    # Step 5: Smooth with moving average (optional, very light smoothing)
    window_size = 3
    signal_smoothed = np.convolve(signal_denoised, np.ones(window_size)/window_size, mode='same')
    
    return signal_smoothed

def normalize_signal(signal_data):
    """
    Normalize signal to 0-1 range for better visualization
    """
    signal_min = np.min(signal_data)
    signal_max = np.max(signal_data)
    if signal_max - signal_min > 0:
        return (signal_data - signal_min) / (signal_max - signal_min)
    return signal_data

# --- Plotting Function ---
def generate_and_save_plot(data, filename="ecg_report.png"):
    if not data or len(data) < 100:
        print("Insufficient data to plot.")
        return None

    # Apply denoising
    print("Applying signal denoising filters...")
    denoised_data = denoise_ecg_signal(data, fs=SAMPLE_RATE)
    
    # Create time axis
    time_axis = np.arange(len(data)) / SAMPLE_RATE

    # Scale denoised signal back to ADC range for consistency
    denoised_scaled = (denoised_data - np.min(denoised_data)) / (np.max(denoised_data) - np.min(denoised_data)) * 4095
    
    # Calculate signal range with extra padding for better visualization
    signal_min = np.min(denoised_scaled)
    signal_max = np.max(denoised_scaled)
    signal_range = signal_max - signal_min
    y_padding = signal_range * 0.3  # 30% padding on top and bottom
    
    # Create figure - denoised signal only
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot denoised signal with medical ECG red color
    ax.plot(time_axis, denoised_scaled, color='#DC143C', linewidth=1.2)
    
    # Set title and labels
    ax.set_title(f"ECG Signal ({PLOT_DURATION_SECONDS}s)", 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Amplitude (mV)", fontsize=13, fontweight='bold')
    
    # Add darker, more visible grid like medical ECG paper
    # Major grid (dark lines every 0.5s and larger voltage intervals)
    ax.grid(True, which='major', linestyle='-', linewidth=1.2, color='#D32F2F', alpha=0.6)
    
    # Minor grid (lighter lines for finer divisions)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.6, color='#EF5350', alpha=0.4)
    
    # Set y-axis limits with extra space (like medical ECG paper)
    ax.set_ylim(signal_min - y_padding, signal_max + y_padding)
    
    # Set background to light pink/cream (like ECG paper)
    ax.set_facecolor('#FFF5F5')
    fig.patch.set_facecolor('white')
    
    # Increase tick label size for better readability
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add more ticks for better grid density (like ECG paper)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))  # Major tick every 0.5s
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Minor tick every 0.1s
    
    # Set y-axis ticks based on signal range
    y_range = (signal_max + y_padding) - (signal_min - y_padding)
    y_major_interval = y_range / 8  # Approximately 8 major divisions
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(y_major_interval / 5))
    
    plt.tight_layout()
    
    # Save the plot with high quality
    full_path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved denoised ECG report to {full_path}")
    
    return full_path

# --- Serial Port Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✓ Connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
    print(f"✓ Collecting {PLOT_DURATION_SECONDS}s segments at {SAMPLE_RATE}Hz")
    print(f"✓ Output folder: {OUTPUT_FOLDER}/")
    print("="*60)
except serial.SerialException as e:
    print(f"✗ Error: Could not open serial port {SERIAL_PORT}. {e}")
    print("\nTroubleshooting:")
    print("1. Ensure ESP32 is connected")
    print("2. Close other programs using the port (Serial Monitor, Plotter)")
    print("3. Check the correct COM port is selected")
    exit()

# --- Main Loop ---
print("Collecting ECG data... Press Ctrl+C to stop.\n")
try:
    sample_count = 0
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # Look for ">ECG:VALUE" pattern
        match = re.match(r'>ECG:(\d+)', line)
        if match:
            try:
                ecg_value = int(match.group(1))
                ecg_data.append(ecg_value)
                sample_count += 1
                
                # Progress indicator
                if sample_count % 500 == 0:
                    progress = (len(ecg_data) / NUM_SAMPLES_PER_PLOT) * 100
                    print(f"Progress: {progress:.1f}% ({len(ecg_data)}/{NUM_SAMPLES_PER_PLOT} samples)", end='\r')

                # If enough data is collected, generate a plot
                if len(ecg_data) >= NUM_SAMPLES_PER_PLOT:
                    print("\n" + "="*60)
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"ecg_report_{timestamp_str}.png"
                    
                    # Generate and save the plot
                    generated_filepath = generate_and_save_plot(ecg_data, output_filename)
                    
                    print("="*60)
                    print(f"✓ Files ready for AI model upload:")
                    print(f"  - {generated_filepath}")
                    print("="*60 + "\n")
                    
                    # --- Optional: Upload to Hugging Face Space ---
                    # if generated_filepath:
                    #     import requests
                    #     print(f"Uploading to Hugging Face Space...")
                    #     with open(generated_filepath, 'rb') as f:
                    #         files = {'file': f}
                    #         response = requests.post("YOUR_API_ENDPOINT", files=files)
                    #         print(f"Response: {response.status_code}")
                    # ------------------------------------------------
                    
                    # Clear the buffer for the next plot
                    ecg_data = []
                    sample_count = 0
                    
            except ValueError:
                print(f"Could not parse ECG value from line: {line}")
        elif line and not line.startswith('>'):
            # Print other serial messages for debugging
            if "Initialized" in line or "---" not in line:
                print(f"ESP32: {line}")

except KeyboardInterrupt:
    print("\n\n✓ Stopping data collection.")
except Exception as e:
    print(f"\n✗ An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if ser.is_open:
        ser.close()
        print("✓ Serial port closed.")

print("\nSession complete!")
print(f"ECG reports saved in: {OUTPUT_FOLDER}/")
print("Upload the '*_denoised_only.png' files to your AI model.")