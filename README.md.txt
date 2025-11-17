# ECG Analysis System 🫀

An automated ECG monitoring system that collects data from ESP32 with AD8232 sensor, analyzes it using deep learning, and generates professional PDF reports.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 🌟 Features

- **Real-time ECG Data Collection** from ESP32 (COM5, 115200 baud)
- **Signal Denoising** using Butterworth bandpass filter
- **AI-Powered Classification** using CNN (Normal vs Abnormal)
- **Professional PDF Reports** with diagnosis and confidence scores
- **Fully Automated Workflow** - Just run one command!

## 📋 Prerequisites

- Python 3.8 or higher
- ESP32 with AD8232 ECG sensor module
- Training dataset (Normal and Abnormal ECG images)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ecg-analysis-system.git
cd ecg-analysis-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure paths (if different from default)

Edit the paths in `main.py` if your dataset location is different:
```python
ecg_reports_dir = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports"
```

## 📁 Project Structure

```
ecg-analysis-system/
├── main.py              # Main orchestrator (run this!)
├── ecgplotter.py        # ECG data collection from ESP32
├── bot.py               # CNN model architecture
├── input.py             # Image processing and inference
├── report.py            # PDF report generation
├── train_model.py       # Model training script (optional)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT License
└── .gitignore          # Git ignore rules
```

## 🎯 Usage

### Quick Start (Automated Workflow)

Simply run the main script:
```bash
python main.py
```

This will automatically:
1. ✅ Connect to ESP32 on COM5 (115200 baud)
2. ✅ Collect 5 seconds of ECG data
3. ✅ Apply denoising filter
4. ✅ Generate ECG plot
5. ✅ Analyze with AI model
6. ✅ Create professional PDF report

### Output Files

All outputs are saved in `ecg_reports/` folder:
- `new_image.png` - Denoised ECG graph
- `new_image.pdf` - Complete analysis report with diagnosis
- `ecg_report_YYYYMMDD_HHMMSS.png` - Timestamped backup copy

### Training Your Own Model (Optional)

If you have training data:

```bash
python train_model.py
```

Required folder structure for training:
```
heart_dataset/
├── Normal/          # Normal ECG images (PNG format)
├── Abnormal/        # Abnormal ECG images (PNG format)
└── ecg_reports/     # Output folder (auto-created)
```

Recommended: At least 100 images per class (200 total) for decent results.

## 🔧 Hardware Setup

### ESP32 with AD8232 Connections:
```
AD8232 Pin      →    ESP32 Pin
──────────────────────────────
OUTPUT          →    GPIO34 (ADC1_CH6)
LO+             →    GPIO26
LO-             →    GPIO27
VCC (3.3V)      →    3.3V
GND             →    GND
```

### ESP32 Arduino Code:

Upload this to your ESP32:

```cpp
// ECG Data Sender for ESP32
const int ecgPin = 34;      // AD8232 OUTPUT to GPIO34
const int loPlus = 26;      // LO+ to GPIO26
const int loMinus = 27;     // LO- to GPIO27

void setup() {
  Serial.begin(115200);
  pinMode(ecgPin, INPUT);
  pinMode(loPlus, INPUT);
  pinMode(loMinus, INPUT);
}

void loop() {
  // Check if electrodes are connected
  if (digitalRead(loPlus) == 1 || digitalRead(loMinus) == 1) {
    Serial.println(">ECG:0");  // No signal
  } else {
    int ecgValue = analogRead(ecgPin);
    Serial.print(">ECG:");
    Serial.println(ecgValue);
  }
  delay(5);  // 200Hz sampling rate
}
```

### Electrode Placement:
- **RA (Right Arm)**: Right side below collarbone
- **LA (Left Arm)**: Left side below collarbone  
- **RL (Right Leg)**: Right lower abdomen/hip

## 🧪 Testing Individual Components

Test each module separately:

```bash
# Test ECG data collection only
python ecgplotter.py

# Test model architecture
python bot.py

# Test image processing
python input.py

# Test PDF generation
python report.py
```

## ⚙️ Configuration

### Serial Port Settings (`ecgplotter.py`)
```python
SERIAL_PORT = 'COM5'           # Change to your port
BAUD_RATE = 115200             # ESP32 baud rate
SAMPLE_RATE = 200              # Hz (samples per second)
PLOT_DURATION_SECONDS = 5      # Data collection duration
```

### Model Settings (`input.py`)
```python
target_size = (224, 224)       # Input image size
batch_size = 32                # Training batch size
num_epochs = 10                # Training epochs
learning_rate = 0.001          # Learning rate
```

### Filter Settings (`ecgplotter.py`)
```python
lowcut = 0.5                   # High-pass filter (Hz)
highcut = 40                   # Low-pass filter (Hz)
```

## 📊 Model Architecture

**ECGModel (Convolutional Neural Network)**

```
Input: Grayscale ECG Image (224x224x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPool2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPool2D (2x2)
    ↓
Flatten → Dense (512) + ReLU + Dropout (0.5)
    ↓
Dense (2) → Softmax
    ↓
Output: [Normal probability, Abnormal probability]
```

**Parameters:**
- Total Parameters: ~12.8M
- Trainable Parameters: ~12.8M
- Input Shape: (Batch, 1, 224, 224)
- Output Shape: (Batch, 2)

## 📈 Performance

Expected performance with sufficient training data:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Inference Time**: ~50-100ms per image (CPU)
- **Data Collection Time**: 5 seconds
- **Total Pipeline Time**: ~6-8 seconds

## 🐛 Troubleshooting

### Problem: Serial Port Not Found
```bash
# Windows - List COM ports
mode

# Linux/Mac - List serial devices
ls /dev/tty*

# Solution: Update SERIAL_PORT in ecgplotter.py
```

### Problem: Model Architecture Mismatch
```bash
# Error: "Missing key(s) in state_dict"
# Solution: Delete old model and retrain
del ecg_model.pth
python train_model.py
```

### Problem: PDF Generation Error (Unicode)
```
# Already fixed in report.py
# Replaced bullet points (•) with dashes (-)
```

### Problem: No ECG Signal
- Check electrode connections
- Ensure electrodes have good skin contact
- Use electrode gel if available
- Check ESP32 serial connection
- Verify correct COM port and baud rate

### Problem: Missing Dependencies
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install torch torchvision numpy pandas scikit-learn pillow scipy matplotlib fpdf pyserial
```

## 🔒 Privacy & Security

- **No data uploaded**: All processing is local
- **No cloud storage**: Data stays on your computer
- **No telemetry**: No tracking or analytics
- **Open source**: Full code transparency

## 📝 Important Notes

### Medical Disclaimer
⚠️ **THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This system is NOT intended for:
- Medical diagnosis
- Clinical decision-making
- Treatment planning
- Emergency medical situations

**Always consult qualified healthcare professionals for medical advice.**

### Limitations
- Model accuracy depends on training data quality and quantity
- Not tested on clinical datasets
- Not FDA approved or medically certified
- Results may vary based on signal quality
- Designed for demonstration purposes

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Add more ECG classification classes (Arrhythmia, MI, etc.)
- [ ] Improve model architecture
- [ ] Add real-time monitoring dashboard
- [ ] Support for multiple ECG leads
- [ ] Cloud storage integration (optional)
- [ ] Mobile app integration
- [ ] Better signal processing algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **FPDF Library** - PDF generation
- **SciPy Team** - Signal processing tools
- **AD8232 Module** - ECG sensor hardware
- **ESP32 Community** - Microcontroller support

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 📞 Support

Having issues? Here's how to get help:

1. **Check Documentation**: Read this README thoroughly
2. **Search Issues**: Look through existing GitHub issues
3. **Open an Issue**: Create a new issue with details
4. **Email**: Contact via email for urgent matters

## 🌟 Star This Repository

If you find this project useful, please consider giving it a ⭐ on GitHub!

## 📚 References

- [ECG Signal Processing](https://en.wikipedia.org/wiki/Electrocardiography)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [AD8232 Datasheet](https://www.analog.com/en/products/ad8232.html)
- [ESP32 Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/)

## 🔮 Future Roadmap

- [ ] Web-based interface with Flask/Streamlit
- [ ] Real-time ECG monitoring dashboard
- [ ] Support for 12-lead ECG
- [ ] Mobile app (Android/iOS)
- [ ] Cloud sync (optional)
- [ ] Multi-class classification (AF, MI, VT, etc.)
- [ ] Heart rate and HRV analysis
- [ ] Export to DICOM format
- [ ] Integration with hospital systems (HL7/FHIR)

---

**Made with ❤️ for the Healthcare & AI Community**

⚠️ **Remember**: This is a research/educational tool. Always consult healthcare professionals for medical decisions.