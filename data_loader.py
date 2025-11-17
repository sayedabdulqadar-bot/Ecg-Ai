import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ECGImageDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.samples = []

        for idx, cls in enumerate(class_names):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.exists(cls_folder):
                continue
            for file in os.listdir(cls_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
