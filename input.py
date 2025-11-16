"""
ECG Image Processing Module
Loads model, processes ECG images, and generates reports
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from bot import ECGModel
from report import generate_pdf


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_data(image_paths, labels, target_size=(224, 224)):
    """Load images from paths and convert to numpy arrays."""
    images = []
    valid_labels = []
    
    for img_path, label in zip(image_paths, labels):
        img_array = preprocess_image(img_path, target_size)
        if img_array is not None:
            images.append(img_array)
            valid_labels.append(label)
    
    images = np.array(images)
    images = np.expand_dims(images, axis=1)  # Add channel dimension: (N, 1, H, W)
    labels = np.array(valid_labels)
    
    return train_test_split(images, labels, test_size=0.2, random_state=42)


def create_dataloader(X, y, batch_size=32):
    """Create PyTorch DataLoader."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_images_from_folders(normal_folder, abnormal_folder, file_extension="*.png"):
    """Load image paths and labels from normal and abnormal folders."""
    normal_image_paths = glob.glob(os.path.join(normal_folder, file_extension))
    abnormal_image_paths = glob.glob(os.path.join(abnormal_folder, file_extension))
    
    images = normal_image_paths + abnormal_image_paths
    labels = [0] * len(normal_image_paths) + [1] * len(abnormal_image_paths)
    
    return images, labels


def train_model_simple(model, train_loader, num_epochs=10, learning_rate=0.001):
    """Simple training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")
    
    return model


def predict(model, image):
    """Run inference on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # Add channel dimension if needed
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
        
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        return probabilities.squeeze().cpu().numpy()


def process_ecg_image(input_image_path, model_weights_path=None):
    """Process ECG image and generate report."""
    print(f"\n{'='*60}")
    print("Starting ECG Analysis...")
    print(f"{'='*60}\n")
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"✗ Error: Input image not found at {input_image_path}")
        return
    
    print(f"✓ Input image found: {input_image_path}")
    
    # Load or train model
    model = ECGModel()
    model_loaded = False
    
    if model_weights_path and os.path.exists(model_weights_path):
        try:
            print(f"✓ Attempting to load model from {model_weights_path}")
            model.load_state_dict(torch.load(model_weights_path))
            model_loaded = True
            print("✓ Model weights loaded successfully")
        except Exception as e:
            print(f"\n⚠ Warning: Could not load model weights!")
            print(f"  Error: {str(e)[:100]}...")
            print(f"\n  The saved model has incompatible architecture.")
            print(f"  Solutions:")
            print(f"    1. Delete '{model_weights_path}' and retrain: python train_model.py")
            print(f"    2. Continue with untrained model (predictions will be random)")
            print(f"\n  Continuing with untrained model...\n")
            model_loaded = False
    else:
        print("⚠ No pre-trained model found.")
        print("  Using untrained model (predictions will be random).")
        print("  To train a model, run: python train_model.py\n")
    
    # Preprocess input image
    print("✓ Preprocessing image...")
    image = preprocess_image(input_image_path)
    
    if image is None:
        print("✗ Error preprocessing image")
        return
    
    # Run prediction
    print("✓ Running model inference...")
    if not model_loaded:
        print("  (Note: Using untrained model - results will be random)")
    
    probabilities_2_class = predict(model, image)
    
    # Build 3-class probabilities: Normal, Abnormal, Atrial Fibrillation (0% for AF)
    report_probabilities = [
        float(probabilities_2_class[0]),  # Normal
        float(probabilities_2_class[1]),  # Abnormal
        0.0  # Atrial Fibrillation (not detected by this model)
    ]
    report_class_names = ["Normal", "Abnormal", "Atrial Fibrillation"]
    
    print(f"\n{'='*60}")
    print("Prediction Results:")
    for i, name in enumerate(report_class_names):
        print(f"  {name}: {report_probabilities[i]*100:.2f}%")
    if not model_loaded:
        print("\n  ⚠ Warning: Results are from untrained model (not accurate)")
    print(f"{'='*60}\n")
    
    # Prepare report data
    report_data = {
        "image": input_image_path,
        "probabilities": report_probabilities,
        "class_names": report_class_names
    }
    
    # Generate PDF report
    output_pdf_path = os.path.join(
        r"C:\Users\sayed\Documents\heart_dataset\ecg_reports",
        "new_image.pdf"
    )
    
    print("✓ Generating PDF report...")
    generate_pdf(report_data, filename=output_pdf_path)
    print(f"✓ Report saved to: {output_pdf_path}")
    
    print(f"\n{'='*60}")
    print("ECG Analysis Complete!")
    print(f"{'='*60}\n")


def train_and_save_model(normal_folder, abnormal_folder, save_path="ecg_model.pth"):
    """Train model and save weights."""
    print("Loading training data...")
    image_paths, labels = load_images_from_folders(normal_folder, abnormal_folder)
    
    if len(image_paths) == 0:
        print("✗ No training images found!")
        return None
    
    print(f"✓ Found {len(image_paths)} images ({labels.count(0)} Normal, {labels.count(1)} Abnormal)")
    
    X_train, X_test, y_train, y_test = load_data(image_paths, labels)
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    test_loader = create_dataloader(X_test, y_test, batch_size=32)
    
    print("\nStarting model training...")
    model = ECGModel()
    trained_model = train_model_simple(model, train_loader, num_epochs=10)
    
    # Save model
    torch.save(trained_model.state_dict(), save_path)
    print(f"\n✓ Model saved to {save_path}")
    
    return trained_model


if __name__ == "__main__":
    # Configuration
    normal_folder = r"C:\Users\sayed\Documents\heart_dataset\Normal"
    abnormal_folder = r"C:\Users\sayed\Documents\heart_dataset\Abnormal"
    input_image_path = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports\new_image.png"
    model_weights_path = "ecg_model.pth"
    
    # Option 1: Train model first (uncomment to train)
    # print("Training model...")
    # train_and_save_model(normal_folder, abnormal_folder, model_weights_path)
    
    # Option 2: Process ECG image with existing or untrained model
    process_ecg_image(input_image_path, model_weights_path)