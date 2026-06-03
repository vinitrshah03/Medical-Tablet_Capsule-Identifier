import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import time

# --- CONFIGURATION ---
CSV_PATH = "dataset/data/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
IMAGE_DIR = "dataset/data/ePillID_data/classification_data"
MODEL_DIR = "model"

# This is now the "Maximum" classes to look for
MAX_CLASSES = 9804  
EPOCHS = 15 
BATCH_SIZE = 16 
WORKERS = 2 

class PillDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DIR, self.df.iloc[idx]['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.df.iloc[idx]['class_id']
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df_raw = pd.read_csv(CSV_PATH)
    
    # 2. Filter logic: Get up to MAX_CLASSES, or all available if dataset is smaller
    available_classes = df_raw['pilltype_id'].nunique()
    target_num_classes = min(MAX_CLASSES, available_classes)
    
    print(f"Dataset has {available_classes} unique pills. Training on top {target_num_classes}.")

    top_pills = df_raw['pilltype_id'].value_counts().nlargest(target_num_classes).index
    df = df_raw[df_raw['pilltype_id'].isin(top_pills)].copy()
    
    # 3. Create strictly matching Labels and Mapping
    df['class_id'] = df['pilltype_id'].astype('category').cat.codes
    actual_num_classes = df['pilltype_id'].nunique() # This is the number app.py needs
    
    mapping = dict(enumerate(df['pilltype_id'].astype('category').cat.categories))
    
    # Save mapping first to ensure it matches the model we are about to build
    with open(os.path.join(MODEL_DIR, "class_mapping.json"), 'w') as f:
        json.dump(mapping, f)

    # 4. Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = PillDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    # 5. Model Setup - Dynamic output layer
    model = models.mobilenet_v3_large(weights='DEFAULT')
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, actual_num_classes)
    
    print(f"Model initialized with {actual_num_classes} output neurons.")
    
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training Loop
    print(f"Starting Training on {len(df)} images...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for i, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(loader)}], Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} Complete | Avg Loss: {running_loss/len(loader):.4f} ---")

    # 7. Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pill_model.pth"))
    print(f"Success! Model and Mapping synced at {actual_num_classes} classes.")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    train()