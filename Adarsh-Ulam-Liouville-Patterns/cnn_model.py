import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter # Import Counter for label distribution

# --- Part 1: Liouville Function and Ulam Spiral Generation (Your Code) ---

def liouville_lambda(n):
    """Calculates the Liouville function lambda(n)."""
    if n < 1:
        return 0 # Or handle as error, Liouville is for positive integers
    omega = 0
    i = 2
    while i * i <= n:
        while n % i == 0:
            omega += 1
            n //= i
        i += 1
    if n > 1:
        omega += 1
    return 1 if omega % 2 == 0 else -1 # 1 for +1, -1 for -1

def generate_ulam_spiral(size):
    """Generates an Ulam spiral grid with Liouville lambda(n) values."""
    grid = np.zeros((size, size), dtype=int)
    x, y = size // 2, size // 2
    dx, dy = 0, -1
    n = 1
    steps_taken = 0
    segment_length = 1
    turns_in_segment = 0

    coords_to_n = {}

    while n <= size * size:
        if 0 <= y < size and 0 <= x < size:
            coords_to_n[(y, x)] = n
        else:
            break

        n += 1
        steps_taken += 1

        if steps_taken == segment_length:
            steps_taken = 0
            turns_in_segment += 1
            dx, dy = -dy, dx

            if turns_in_segment % 2 == 0:
                segment_length += 1

        x, y = x + dx, y + dy

    for r in range(size):
        for c in range(size):
            if (r, c) in coords_to_n:
                grid[r, c] = liouville_lambda(coords_to_n[(r, c)])
            else:
                grid[r, c] = 0

    return grid

# --- Part 2: Patch Extraction and Labeling ---

def extract_and_label_patches(grid, patch_size, num_patches_per_type=1000):
    """
    Extracts patches from specific regions of the Ulam spiral and labels them.
    Labels: 0 (predominantly red/-1), 1 (predominantly blue/+1), 2 (mixed)
    """
    size = grid.shape[0]
    patches = []
    labels = []

    # --- ADJUST THESE THRESHOLDS ---
    # Lowered to make it easier for patches to be classified as Red or Blue
    red_threshold = 0.55
    blue_threshold = 0.55

    def get_patch(center_y, center_x):
        half_patch = patch_size // 2
        y_start = max(0, center_y - half_patch)
        y_end = min(size, center_y + half_patch + (patch_size % 2))
        x_start = max(0, center_x - half_patch)
        x_end = min(size, center_x + half_patch + (patch_size % 2))

        patch = grid[y_start:y_end, x_start:x_end]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            # Pad if patch is at edge and clipped
            padded_patch = np.zeros((patch_size, patch_size), dtype=grid.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        return patch

    def label_patch(patch):
        red_count = np.sum(patch == -1)
        blue_count = np.sum(patch == 1)
        total_pixels = patch_size * patch_size
        if total_pixels == 0: return 2

        if (red_count / total_pixels) >= red_threshold:
            return 0 # Predominantly Red (-1)
        elif (blue_count / total_pixels) >= blue_threshold:
            return 1 # Predominantly Blue (+1)
        else:
            return 2 # Mixed

    print("Extracting Antidiagonal patches...")
    for _ in range(num_patches_per_type):
        idx = random.randint(0, size - 1)
        center_y = idx
        center_x = size - 1 - idx
        patch = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)
            labels.append(label_patch(patch))

    print("Extracting 'Reverse L' patches...")
    # This is a conceptual region for the "Reverse L" based on your observations
    # It focuses on the top-right corner area, but not the very edge pixel
    corner_start_y = 0
    corner_start_x = size - int(size * 0.15) # Start 15% from right edge
    corner_end_y = int(size * 0.15) # End 15% from top edge
    corner_end_x = size - 1

    for _ in range(num_patches_per_type):
        center_y = random.randint(corner_start_y, corner_end_y)
        center_x = random.randint(corner_start_x, corner_end_x)
        patch = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)
            labels.append(label_patch(patch))

    print("Extracting Random patches...")
    for _ in range(num_patches_per_type):
        center_y = random.randint(0, size - 1)
        center_x = random.randint(0, size - 1)
        patch = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)
            labels.append(label_patch(patch))

    patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)

    return patches_tensor, labels_tensor

# --- Part 3: Simple CNN Model for Patch Classification ---

class SimpleCNN(nn.Module):
    def __init__(self, patch_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = (patch_size // 4) * (patch_size // 4) * 32
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Part 4: Training and Evaluation Loop ---

def train_and_evaluate_model(patches, labels, patch_size, num_epochs=10, batch_size=32):
    num_classes = 3 # Explicitly set to 3 for our 0, 1, 2 labels

    model = SimpleCNN(patch_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(patches, labels)

    # --- Print Label Distribution Before Split ---
    label_counts = Counter(labels.numpy())
    print(f"\nLabel distribution before train/test split: {label_counts}")
    label_map_full = {0: 'Red (-1)', 1: 'Blue (+1)', 2: 'Mixed'}
    for label_val, count in label_counts.items():
        print(f"  {label_map_full[label_val]}: {count} samples")


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # --- Check label distribution in test set ---
    test_labels_list = [label.item() for _, label in test_dataset]
    test_label_counts = Counter(test_labels_list)
    print(f"\nLabel distribution in TEST set: {test_label_counts}")
    for label_val, count in test_label_counts.items():
        print(f"  {label_map_full[label_val]}: {count} samples")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTraining Simple CNN on {len(train_dataset)} patches, validating on {len(test_dataset)} patches...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}%")

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            # Ensure c is a 1D tensor for iteration
            if c.dim() == 0: # Handle batch_size=1 case
                c = c.unsqueeze(0)
            if targets.dim() == 0: # Handle batch_size=1 case
                targets = targets.unsqueeze(0)

            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\nAccuracy per class:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f"  {label_map_full[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_total[i]} samples)")
        else:
            print(f"  {label_map_full[i]}: No samples in test set for this class.")

# --- Main Execution ---
if __name__ == "__main__":
    SPIRAL_SIZE = 5001
    PATCH_SIZE = 16
    NUM_PATCHES_PER_TYPE = 5000 # Increased number of patches

    print(f"Generating Ulam spiral of size {SPIRAL_SIZE}x{SPIRAL_SIZE}...")
    ulam_grid = generate_ulam_spiral(SPIRAL_SIZE)
    print("Ulam spiral generated.")

    print(f"Extracting and labeling patches (Patch Size: {PATCH_SIZE}x{PATCH_SIZE})...")
    all_patches, all_labels = extract_and_label_patches(ulam_grid, PATCH_SIZE, NUM_PATCHES_PER_TYPE)
    print(f"Total patches extracted: {len(all_patches)}")

    # Corrected cmap for visualization to 'bwr' for Red/Blue/White
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    sample_indices = random.sample(range(len(all_patches)), 6)
    label_map_full = {0: 'Red', 1: 'Blue', 2: 'Mixed'}
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        # Use 'bwr' colormap, and set vmin/vmax to ensure -1 is one extreme, 1 is the other
        ax.imshow(all_patches[idx].squeeze().numpy(), cmap='bwr', vmin=-1, vmax=1, origin='lower')
        ax.set_title(f"Label: {label_map_full[all_labels[idx].item()]}")
        ax.axis('off')
    plt.suptitle("Sample Patches")
    plt.tight_layout()
    plt.show()

    train_and_evaluate_model(all_patches, all_labels, PATCH_SIZE, num_epochs=15, batch_size=64)