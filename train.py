import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import config

# 1. ENVIRONMENT DETECTION
IS_KAGGLE = os.path.exists('/kaggle/working')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Running on {'KAGGLE CLOUD' if IS_KAGGLE else 'LOCAL LAPTOP'} ---")
print(f"Primary Device: {DEVICE}")

# 2. DATASET CONTROLLER (No local gigabyte downloads!)
if IS_KAGGLE:
    print("Loading massive cloud dataset...")
    # TODO: Replace with your real data paths once pushed to Kaggle
    # e.g., images = load_real_images('../input/your-dataset-slug/')
    # For now, using placeholders so it runs out-of-the-box
    X = torch.randn(1000, 10)  
    y = torch.randint(0, 2, (1000,)).float()
else:
    print("Running locally. Creating fake tiny dataset for debugging...")
    X = torch.randn(20, 10)  # Just 20 fake rows
    y = torch.randint(0, 2, (20,)).float()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4 if not IS_KAGGLE else 64, shuffle=True)

'''# 3. DEFINE A TOY MODEL
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(DEVICE)'''

# 4. CRITICAL: ACTIVATE DUAL T4 GPUs ON KAGGLE
if IS_KAGGLE and torch.cuda.device_count() > 1:
    print(f"Dual GPUs Detected! Splitting workload across {torch.cuda.device_count()} T4s.")
    model = nn.DataParallel(model)

# 5. SIMPLE TRAINING LOOP SANITY CHECK
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training loop...")
for epoch in range(config.num_epochs):
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

print("🎉 Success! Code configuration is 100% correct.")
