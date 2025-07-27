import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter

# --- Part 1: Liouville Function and Ulam Spiral Generation ---

def liouville_lambda(n):
    """
    Calculates the Liouville function lambda(n).
    Returns 1 for lambda(n)=+1, -1 for lambda(n)=-1.
    """
    if n < 1:
        return 0
    omega = 0
    i = 2
    while i * i <= n:
        while n % i == 0:
            omega += 1
            n //= i
        i += 1
    if n > 1:
        omega += 1
    return 1 if omega % 2 == 0 else -1

def generate_ulam_spiral_and_numbers(size):
    """
    Generates an Ulam spiral grid with Liouville lambda(n) values
    and a separate grid with the actual n values.
    It also returns a mapping from n to its (y, x) coordinates.

    Args:
        size (int): The side length of the square Ulam spiral grid.

    Returns:
        tuple: (lambda_grid, n_grid, n_to_coords)
            - lambda_grid (np.ndarray): Grid with lambda(n) values (-1 or 1).
            - n_grid (np.ndarray): Grid with actual integer 'n' values.
            - n_to_coords (dict): Maps integer 'n' to its (y, x) coordinate.
    """
    lambda_grid = np.zeros((size, size), dtype=int)
    n_grid = np.zeros((size, size), dtype=int)
    n_to_coords = {}

    x, y = size // 2, size // 2
    dx, dy = 0, -1
    n = 1
    steps_taken = 0
    segment_length = 1
    turns_in_segment = 0

    coords_to_n = {} # Maps (y, x) to n (temporary for spiral generation)

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

    # Populate grids and n_to_coords from coords_to_n
    for r in range(size):
        for c in range(size):
            if (r, c) in coords_to_n:
                current_n = coords_to_n[(r, c)]
                n_grid[r, c] = current_n
                lambda_grid[r, c] = liouville_lambda(current_n)
                n_to_coords[current_n] = (r, c)
            # else: n_grid[r, c] and lambda_grid[r, c] remain 0 (default)

    return lambda_grid, n_grid, n_to_coords

# --- Part 2: Patch Extraction and Labeling for PIXEL PREDICTION ---

def extract_and_label_patches_for_pixel_prediction(grid, patch_size, num_patches_per_type=1000):
    """
    Extracts patches and labels them with the lambda(n) value of their CENTRAL pixel.
    Labels: 0 (for -1, i.e., Red), 1 (for +1, i.e., Blue).
    Mixed patches are NOT used here, as we predict a specific pixel's value.

    Args:
        grid (np.ndarray): The Ulam spiral grid with lambda(n) values (-1 or 1).
        patch_size (int): The side length of the square patches to extract.
        num_patches_per_type (int): Number of patches to extract for each category
                                    (Antidiagonal, 'Reverse L', Random).

    Returns:
        tuple: (patches_tensor, labels_tensor)
            - patches_tensor (torch.Tensor): Tensor of extracted image patches.
            - labels_tensor (torch.Tensor): Tensor of corresponding labels.
    """
    size = grid.shape[0]
    patches = []
    labels = [] # Will be 0 (for -1) or 1 (for +1)

    def get_patch(center_y, center_x):
        """Safely extracts a patch from the grid, padding if at the edge."""
        half_patch = patch_size // 2
        y_start = max(0, center_y - half_patch)
        y_end = min(size, center_y + half_patch + (patch_size % 2))
        x_start = max(0, center_x - half_patch)
        x_end = min(size, center_x + half_patch + (patch_size % 2))

        patch = grid[y_start:y_end, x_start:x_end]
        
        # Pad if patch is at edge and clipped (ensure it's patch_size x patch_size)
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            padded_patch = np.zeros((patch_size, patch_size), dtype=grid.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        return patch, (center_y, center_x) # Return patch and its true center (needed for grid lookup)

    # We sample from various regions to ensure diverse central pixels.
    # The 'num_patches_per_type' now represents the number of patches from *each* category.

    # 1. Antidiagonal Patches
    print("Extracting Antidiagonal patches for central pixel prediction...")
    for _ in range(num_patches_per_type):
        center_y = random.randint(0, size - 1)
        center_x = size - 1 - center_y # Point on antidiagonal
        
        patch, (true_center_y, true_center_x) = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size): # Only add if successfully extracted full patch
            # Get the lambda value of the true central pixel of the patch
            # Ensure true_center_y and true_center_x are within original grid bounds for grid lookup
            if 0 <= true_center_y < size and 0 <= true_center_x < size:
                central_pixel_lambda = grid[true_center_y, true_center_x]
                patches.append(patch)
                labels.append(0 if central_pixel_lambda == -1 else 1) # Map -1 to 0, +1 to 1
            else:
                continue # Skip if true center falls outside original grid (due to padding at spiral edges)

    # 2. "Reverse L" Region Patches
    print("Extracting 'Reverse L' patches for central pixel prediction...")
    # This is a conceptual region for the "Reverse L" based on your observations
    # It focuses on the top-right corner area.
    corner_start_y = 0
    corner_start_x = size - int(size * 0.15)
    corner_end_y = int(size * 0.15)
    corner_end_x = size - 1

    for _ in range(num_patches_per_type):
        # Randomly sample within the top-right corner region
        center_y = random.randint(corner_start_y, corner_end_y)
        center_x = random.randint(corner_start_x, corner_end_x)
        
        patch, (true_center_y, true_center_x) = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size):
            if 0 <= true_center_y < size and 0 <= true_center_x < size:
                central_pixel_lambda = grid[true_center_y, true_center_x]
                patches.append(patch)
                labels.append(0 if central_pixel_lambda == -1 else 1)
            else:
                continue

    # 3. Random Background Patches
    print("Extracting Random patches for central pixel prediction...")
    for _ in range(num_patches_per_type):
        center_y = random.randint(0, size - 1)
        center_x = random.randint(0, size - 1)
        
        patch, (true_center_y, true_center_x) = get_patch(center_y, center_x)
        if patch.shape == (patch_size, patch_size):
            if 0 <= true_center_y < size and 0 <= true_center_x < size:
                central_pixel_lambda = grid[true_center_y, true_center_x]
                patches.append(patch)
                labels.append(0 if central_pixel_lambda == -1 else 1)
            else:
                continue

    patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1) # Add channel dimension
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)

    return patches_tensor, labels_tensor

# --- Part 3: Simple CNN Model for PIXEL PREDICTION ---

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for 1-pixel classification."""
    def __init__(self, patch_size, num_classes=2): # num_classes is now 2 (-1 or 1)
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Input channel 1 (grayscale)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input features for the linear layer after convolutions and pooling
        self.flattened_size = (patch_size // 4) * (patch_size // 4) * 32
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Output 2 classes (0 or 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten the tensor for the fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Part 4: Training and Evaluation Loop for PIXEL PREDICTION ---

def train_and_evaluate_model(patches, labels, patch_size, num_epochs=15, batch_size=64):
    """
    Trains and evaluates the SimpleCNN model for 1-pixel prediction.

    Args:
        patches (torch.Tensor): Tensor of image patches.
        labels (torch.Tensor): Tensor of corresponding labels (0 for -1, 1 for +1).
        patch_size (int): Size of the image patches.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    num_classes = 2 # Explicitly set to 2 for our 0 (-1) and 1 (+1) labels

    model = SimpleCNN(patch_size, num_classes)
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is suitable for 2 classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(patches, labels)

    label_counts = Counter(labels.numpy())
    print(f"\nLabel distribution before train/test split: {label_counts}")
    label_map_full = {0: 'Red (-1)', 1: 'Blue (+1)'} # Only 2 labels now
    for label_val, count in label_counts.items():
        print(f"  {label_map_full[label_val]}: {count} samples")


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

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
            if c.dim() == 0:
                c = c.unsqueeze(0)
            if targets.dim() == 0:
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
# This __main__ block is for local testing/demonstration of the 1-pixel prediction CNN.
if __name__ == "__main__":
    SPIRAL_SIZE = 5001 # Example size for local testing
    PATCH_SIZE = 16
    NUM_PATCHES_PER_TYPE = 5000 # Increased number of patches

    print(f"Generating Ulam spiral of size {SPIRAL_SIZE}x{SPIRAL_SIZE}...")
    # Generate grids using the unified generator from ulam_utils
    # generate_ulam_spiral_and_numbers returns lambda_grid, n_grid, n_to_coords
    ulam_grid_lambda, _, _ = generate_ulam_spiral_and_numbers(SPIRAL_SIZE)
    print("Ulam spiral generated.")

    print(f"Extracting and labeling patches (Patch Size: {PATCH_SIZE}x{PATCH_SIZE})...")
    # Use the new extract_and_label_patches_for_pixel_prediction function
    all_patches, all_labels = extract_and_label_patches_for_pixel_prediction(ulam_grid_lambda, PATCH_SIZE, NUM_PATCHES_PER_TYPE)
    print(f"Total patches extracted: {len(all_patches)}")

    # Visualize some sample patches
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    sample_indices = random.sample(range(len(all_patches)), 6)
    label_map_viz = {0: 'Red', 1: 'Blue'} # Visual map for 2 classes
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        ax.imshow(all_patches[idx].squeeze().numpy(), cmap='bwr', vmin=-1, vmax=1, origin='lower')
        ax.set_title(f"Label: {label_map_viz[all_labels[idx].item()]}")
        ax.axis('off')
    plt.suptitle("Sample Patches for 1-Pixel Prediction Training")
    plt.tight_layout()
    plt.show()

    train_and_evaluate_model(all_patches, all_labels, PATCH_SIZE, num_epochs=15, batch_size=64)