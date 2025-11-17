# test_loader_run.py
from data_loader import get_loaders
data_dir = r"C:\Users\sayed\Documents\heart_dataset\heart_dataset_split"
train_loader, test_loader, classes = get_loaders(data_dir, batch_size=8)
print("Loaded ok. Classes:", classes)
