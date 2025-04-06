import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

data = np.load("indianpinearray.npy")
labels = np.load("IPgt.npy")

data = (data - np.min(data)) / (np.max(data) - np.min(data))

H, W, C = data.shape
X = data.reshape(-1, C)
y = labels.reshape(-1)

mask = y > 0
X = X[mask]
y = y[mask] - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class HyperspectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

train_dataset = HyperspectralDataset(X_train_tensor, y_train_tensor)
test_dataset = HyperspectralDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class PixelCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PixelCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * in_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_model(model, loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

def compute_ndvi(image, nir_idx, red_idx):
    nir = image[:, :, nir_idx]
    red = image[:, :, red_idx]
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def compute_lci(image, nir_idx, green_idx):
    nir = image[:, :, nir_idx]
    green = image[:, :, green_idx]
    lci = (nir - green) / (nir + green + 1e-8)
    return lci

def predict_full_image(model, data_image, mask_valid):
    model.eval()
    H, W, C = data_image.shape
    flat_data = data_image[mask_valid].reshape(-1, C)
    flat_tensor = torch.tensor(flat_data, dtype=torch.float32)

    with torch.no_grad():
        flat_tensor = flat_tensor.unsqueeze(1)
        flat_tensor = flat_tensor.permute(0, 2, 1)
        preds = model(flat_tensor.squeeze(-1))
        preds = torch.argmax(preds, dim=1).cpu().numpy()

    full_pred = np.zeros((H * W), dtype=np.uint8)
    flat_mask = mask_valid.flatten()
    full_pred[flat_mask] = preds
    return full_pred.reshape(H, W)

def visualize_overlay(ndvi, lci, prediction_map):
    vegetation_mask = prediction_map == 0

    ndvi_veg = np.zeros_like(ndvi)
    ndvi_veg[vegetation_mask] = ndvi[vegetation_mask]

    lci_veg = np.zeros_like(lci)
    lci_veg[vegetation_mask] = lci[vegetation_mask]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(prediction_map, cmap="jet")
    axs[0].set_title("Predicted Classes")

    im1 = axs[1].imshow(ndvi_veg, cmap="YlGn")
    axs[1].set_title("NDVI (vegetation only)")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(lci_veg, cmap="YlOrRd")
    axs[2].set_title("LCI (vegetation only)")
    plt.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("visualization_output.png")
    plt.show()

if __name__ == "__main__":
    num_classes = len(np.unique(y))
    model = PixelCNN(in_channels=C, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=100)
    evaluate(model, test_loader)

    ndvi = compute_ndvi(data, nir_idx=48, red_idx=29)
    lci = compute_lci(data, nir_idx=48, green_idx=19)
    print("NDVI shape:", ndvi.shape)
    print("LCI shape:", lci.shape)

    valid_mask = labels > 0
    pred_map = predict_full_image(model, data, valid_mask)

    visualize_overlay(ndvi, lci, pred_map)