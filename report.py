"""
ECG Report Generator
Generates professional PDF reports from ECG analysis results
Output: new_image.pdf in ecg_reports folder
"""

import os
from fpdf import FPDF
from datetime import datetime


class PDF(FPDF):
    """Custom PDF class for ECG reports."""
    
    def header(self):
        """Add a header to the PDF."""
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, txt="ECG Monitoring Report", ln=True, align='C')
        self.ln(5)

    def footer(self):
        """Add a footer with page number."""
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def add_report_info(self):
        """Add report generation timestamp."""
        self.set_font("Arial", 'I', 10)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, txt=f"Generated: {timestamp}", ln=True, align='R')
        self.ln(5)

    def add_ecg_image(self, image_path):
        """Add ECG image to the PDF."""
        try:
            if os.path.exists(image_path):
                # Get page width and set image width
                page_width = self.w - 20  # 10mm margin on each side
                self.image(image_path, x=10, y=self.get_y(), w=page_width)
                self.ln(60)  # Space after image
            else:
                self.set_font("Arial", 'I', 10)
                self.cell(0, 10, txt=f"[Image not found: {image_path}]", ln=True, align='C')
                self.ln(10)
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
            self.set_font("Arial", 'I', 10)
            self.cell(0, 10, txt=f"[Could not load image: {str(e)}]", ln=True, align='C')
            self.ln(10)

    def add_prediction(self, probabilities, class_names):
        """Add the prediction results to the PDF."""
        self.ln(10)
        
        # Title
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, txt="Analysis Results:", ln=True)
        self.ln(3)
        
        # Probability breakdown (using simple dash instead of bullet)
        self.set_font("Arial", size=12)
        for i, name in enumerate(class_names):
            percent = probabilities[i] * 100
            self.cell(0, 8, txt=f"  - {name}: {percent:.2f}%", ln=True)
        
        self.ln(5)
        
        # Final diagnosis (based on first two classes: Normal vs Abnormal)
        prediction_index = 0 if probabilities[0] > probabilities[1] else 1
        final_prediction = class_names[prediction_index]
        confidence = probabilities[prediction_index] * 100
        
        self.set_font("Arial", 'B', 13)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, txt=f"Diagnosis: {final_prediction} ({confidence:.2f}% confidence)", 
                  ln=True, align='C', fill=True)
        
        self.ln(5)
        
        # Medical disclaimer
        self.set_font("Arial", 'I', 9)
        self.set_text_color(100, 100, 100)
        disclaimer = ("Note: This is an automated analysis and should not replace "
                     "professional medical evaluation. Please consult a healthcare provider "
                     "for proper diagnosis and treatment.")
        self.multi_cell(0, 5, txt=disclaimer, align='C')
        self.set_text_color(0, 0, 0)

    def save_pdf(self, filename):
        """Output the PDF to a file."""
        # Ensure directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.output(filename)


def generate_pdf(report_data, filename="ECG_report.pdf"):
    """
    Generate a PDF report from ECG analysis.
    
    Args:
        report_data (dict): Dictionary with keys:
            - 'image': Path to ECG image
            - 'probabilities': List of probabilities for each class [Normal, Abnormal, AF]
            - 'class_names': List of class names
        filename (str): Output PDF filename (full path)
    
    Returns:
        str: Path to generated PDF file, or None if error
    """
    try:
        print(f"\nGenerating PDF report...")
        
        # Create PDF instance
        pdf = PDF()
        pdf.add_page()
        
        # Add timestamp
        pdf.add_report_info()
        
        # Add ECG image
        if 'image' in report_data:
            print(f"  - Adding ECG image: {report_data['image']}")
            pdf.add_ecg_image(report_data["image"])
        
        # Add prediction results
        if 'probabilities' in report_data and 'class_names' in report_data:
            print(f"  - Adding analysis results")
            pdf.add_prediction(report_data["probabilities"], report_data["class_names"])
        
        # Save the PDF
        pdf.save_pdf(filename)
        print(f"  - PDF saved to: {filename}")
        print(f"✓ PDF report generated successfully!\n")
        
        return filename
        
    except Exception as e:
        print(f"✗ Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the report generation."""
    print("="*60)
    print("Testing PDF Report Generation")
    print("="*60)
    
    # Test data (simulated analysis results)
    test_data = {
        "image": r"C:\Users\sayed\Documents\heart_dataset\ecg_reports\new_image.png",
        "probabilities": [0.85, 0.15, 0.0],  # 85% Normal, 15% Abnormal, 0% AF
        "class_names": ["Normal", "Abnormal", "Atrial Fibrillation"]
    }
    
    # Generate test report
    output_path = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports\test_report.pdf"
    
    print(f"\nTest Configuration:")
    print(f"  Image: {test_data['image']}")
    print(f"  Output: {output_path}")
    print(f"  Probabilities: Normal={test_data['probabilities'][0]*100:.1f}%, "
          f"Abnormal={test_data['probabilities'][1]*100:.1f}%")
    
    result = generate_pdf(test_data, filename=output_path)
    
    if result:
        print("\n" + "="*60)
        print("✓ Test completed successfully!")
        print(f"Check the generated PDF at: {output_path}")
        print("="*60)
    else:
        print("\n✗ Test failed!")