import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"C:\Users\laptech\OneDrive\Desktop\Whale backup\whales_only.csv"
IMAGE_FOLDER = r"C:\Users\laptech\Downloads\train_images"

BATCH_SIZE = 16
EPOCHS = 3
LR = 0.0003

# -----------------------------
# CHECK PATHS
# -----------------------------
print("CSV exists:", os.path.exists(CSV_PATH))
print("Image folder exists:", os.path.exists(IMAGE_FOLDER))

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
print("\nColumns:", df.columns)

# take subset of 1000 for faster training
df = df.sample(1000, random_state=42).reset_index(drop=True)

# auto-detect columns
image_col = [c for c in df.columns if "image" in c.lower()][0]
label_col = [c for c in df.columns if ("species" in c.lower() or "id" in c.lower())][0]

print("Image column:", image_col)
print("Label column:", label_col)

# -----------------------------
# REMOVE MISSING FILES
# -----------------------------
exists_mask = df[image_col].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_FOLDER, x))
)

print("Missing images removed:", (~exists_mask).sum())
df = df[exists_mask].reset_index(drop=True)

# -----------------------------
# LABEL ENCODING
# -----------------------------
labels = df[label_col].unique()
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

print("Total classes:", num_classes)

# -----------------------------
# DATASET CLASS
# -----------------------------
class WhaleDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][image_col]
        label_name = self.df.iloc[idx][label_col]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label = label_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = WhaleDataset(df, IMAGE_FOLDER, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for batch_idx, (images, targets) in enumerate(loader):
        print("Batch:", batch_idx)

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch Loss:", total_loss / len(loader))

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "whale_model.pth")
print("\n✅ Model saved as whale_model.pth")
