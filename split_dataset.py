import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIG ---
SRC = r"C:\Users\sayed\Documents\heart_dataset"   # source dataset path
OUT = r"C:\Users\sayed\Documents\heart_dataset_split"  # output split folder
TEST_RATIO = 0.2  # 20% test set
# ----------------

# Create output structure
os.makedirs(OUT, exist_ok=True)

# Supported image extensions
valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.jfif')

print("\n🔍 Checking dataset folders...\n")

for split in ["train", "test"]:
    os.makedirs(os.path.join(OUT, split), exist_ok=True)

summary = []

# Loop through all class folders
for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)
    if not os.path.isdir(src_cls):
        continue  # skip files at root

    images = [f for f in os.listdir(src_cls) if f.lower().endswith(valid_ext)]

    if len(images) == 0:
        print(f"⚠️ No images found in '{cls}' — skipping this folder.")
        continue

    os.makedirs(os.path.join(OUT, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(OUT, "test", cls), exist_ok=True)

    train, test = train_test_split(images, test_size=TEST_RATIO, random_state=42)

    for f in train:
        shutil.copy(os.path.join(src_cls, f), os.path.join(OUT, "train", cls, f))
    for f in test:
        shutil.copy(os.path.join(src_cls, f), os.path.join(OUT, "test", cls, f))

    summary.append((cls, len(train), len(test)))

# Summary report
print("\n✅ Dataset Split Completed!\n")
print("Summary:")
print("-" * 40)
for cls, train_count, test_count in summary:
    print(f"{cls:25s} | Train: {train_count:4d} | Test: {test_count:4d}")
print("-" * 40)
print(f"📁 Split dataset saved at: {OUT}\n")
