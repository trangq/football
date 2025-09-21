import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class CropDataset(Dataset):
    def __init__(self, crop_dir, label_txt, transform=None):
        self.crop_dir = crop_dir
        self.transform = transform
        self.samples = []
        with open(label_txt, "r") as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, int(label)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.crop_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Cấu hình
num_classes = 12  # 0: không rõ, 1-10, 11: >=11
batch_size = 32
epochs = 10

train_dir = "football_crops/train"
train_label = "football_crops_labels/train.txt"
test_dir = "football_crops/test"
test_label = "football_crops_labels/test.txt"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

train_dataset = CropDataset(train_dir, train_label, transform)
test_dataset = CropDataset(test_dir, test_label, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model đơn giản (ResNet18)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train loop
for epoch in range(epochs):
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_bar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}/{epochs} done.")

# Đánh giá
model.eval()
correct = 0
total = 0
test_bar = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for images, labels in test_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {correct/total:.2%}")

torch.save(model.state_dict(), "shirt_number_classifier.pt")