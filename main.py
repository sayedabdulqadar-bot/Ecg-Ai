"""
Main Orchestrator for ECG Analysis System

This script automatically runs the complete workflow:
1. Collect ECG data from ESP32 sensor (COM5) - ecgplotter.py
2. Process the generated image with AI model - input.py
3. Generate PDF report - report.py

Usage:
    python main.py
"""

import os
import sys



def check_dependencies():
    """Check if all required modules are available."""
    required_modules = [
        'torch', 'numpy', 'matplotlib', 'PIL', 
        'sklearn', 'scipy', 'fpdf', 'serial'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"✗ Missing required modules: {', '.join(missing)}")
        print("  Install with: pip install torch numpy matplotlib pillow scikit-learn scipy fpdf pyserial")
        return False
    
    return True


def main():
    """Run the complete ECG analysis workflow automatically."""
    print("\n" + "="*70)
    print(" ECG ANALYSIS SYSTEM - AUTOMATED WORKFLOW")
    print("="*70 + "\n")
    
    # Check dependencies first
    print("Checking dependencies...")
    if not check_dependencies():
        print("\n✗ Please install missing dependencies before running.")
        sys.exit(1)
    
    print("✓ All dependencies available\n")
    
    # Configuration
    ecg_reports_dir = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports"
    input_image_path = os.path.join(ecg_reports_dir, "new_image.png")
    model_weights_path = "ecg_model.pth"
    
    # Ensure output directory exists
    os.makedirs(ecg_reports_dir, exist_ok=True)
    
    # ==========================================
    # STEP 1: Collect ECG Data from ESP32
    # ==========================================
    print("STEP 1: Collecting ECG Data from ESP32 (COM5)")
    print("-" * 70)
    
    try:
        from ecgplotter import collect_ecg_data
        output_image = collect_ecg_data()
        
        if not os.path.exists(output_image):
            print(f"✗ Error: ECG image not generated at {output_image}")
            sys.exit(1)
        
        print(f"\n✓ ECG data collection complete!")
        print(f"  Image saved: {output_image}\n")
        
    except Exception as e:
        print(f"✗ Error during ECG data collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==========================================
    # STEP 2: Process ECG Image with AI Model
    # ==========================================
    print("\n" + "="*70)
    print("STEP 2: Processing ECG Image with AI Model")
    print("-" * 70)
    
    try:
        from input import process_ecg_image
        process_ecg_image(input_image_path, model_weights_path)
        
        print(f"\n✓ ECG image processing complete!\n")
        
    except Exception as e:
        print(f"✗ Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==========================================
    # STEP 3: Verify Report Generation
    # ==========================================
    print("\n" + "="*70)
    print("STEP 3: Verifying Report Generation")
    print("-" * 70)
    
    output_pdf = os.path.join(ecg_reports_dir, "new_image.pdf")
    
    if os.path.exists(output_pdf):
        print(f"✓ PDF report successfully generated!")
        print(f"  Location: {output_pdf}")
    else:
        print(f"✗ Warning: PDF report not found at {output_pdf}")
        sys.exit(1)
    
    # ==========================================
    # WORKFLOW COMPLETE
    # ==========================================
    print("\n" + "="*70)
    print(" ✓ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated Files:")
    print(f"  • ECG Image: {input_image_path}")
    print(f"  • PDF Report: {output_pdf}")
    print(f"\nAll files saved in: {ecg_reports_dir}")
    print("="*70 + "\n")
    
    print("You can now view the report or upload it to your system.")
    print("To run again, simply execute: python main.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error in main workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)