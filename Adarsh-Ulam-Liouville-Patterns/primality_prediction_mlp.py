import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter

# --- Part 1: Helper Functions (Self-Contained) ---

def is_prime(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def liouville_lambda(n): # Keeping this here for completeness, though not directly used for primality labels
    if n < 1: return 0
    omega = 0
    i = 2
    while i * i <= n:
        while n % i == 0: omega += 1; n //= i
        i += 1
    if n > 1: omega += 1
    return 1 if omega % 2 == 0 else -1

def generate_ulam_spiral_and_numbers(size):
    n_grid = np.zeros((size, size), dtype=int)
    n_to_coords = {}
    x, y = size // 2, size // 2; dx, dy = 0, -1; n = 1; steps_taken = 0; segment_length = 1; turns_in_segment = 0
    coords_to_n = {}
    while n <= size * size:
        if 0 <= y < size and 0 <= x < size: coords_to_n[(y, x)] = n
        else: break
        n += 1; steps_taken += 1
        if steps_taken == segment_length:
            steps_taken = 0; turns_in_segment += 1; dx, dy = -dy, dx
            if turns_in_segment % 2 == 0: segment_length += 1
        x, y = x + dx, y + dy
    for r in range(size):
        for c in range(size):
            if (r, c) in coords_to_n:
                current_n = coords_to_n[(r, c)]; n_grid[r, c] = current_n; n_to_coords[current_n] = (r, c)
            else: n_grid[r, c] = 0
    return n_grid, n_to_coords

# --- Part 2: Data Preparation for Primality Prediction ---

class CoordinateDataset(Dataset):
    def __init__(self, coordinates, labels):
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.coordinates[idx], self.labels[idx]

def get_sinusoidal_positional_encoding(position, d_model):
    position = torch.tensor(position, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(d_model)
    pe[0::2] = torch.sin(position * div_term)
    pe[1::2] = torch.cos(position * div_term)
    return pe

def prepare_primality_data(spiral_size, encoding_dim=128):
    n_grid, _ = generate_ulam_spiral_and_numbers(spiral_size)
    coords = []; labels = []
    max_coord_val = spiral_size - 1; pe_scale_factor = 500.0
    for r in range(spiral_size):
        for c in range(spiral_size):
            n_val = n_grid[r, c]
            if n_val > 0:
                pe_y = get_sinusoidal_positional_encoding(r / max_coord_val * pe_scale_factor, encoding_dim)
                pe_x = get_sinusoidal_positional_encoding(c / max_coord_val * pe_scale_factor, encoding_dim)
                normalized_spiral_size = torch.tensor([spiral_size / 5001.0])
                coords.append(torch.cat((pe_y, pe_x, normalized_spiral_size)).numpy())
                labels.append(1 if is_prime(n_val) else 0)
    return np.array(coords), np.array(labels)

# --- Part 3: Multi-Layer Perceptron (MLP) Model ---

class MLP(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x

# --- Part 4: Training and Evaluation Loop ---

def train_and_evaluate_mlp(coords, labels, num_epochs=50, batch_size=512):
    num_classes = 2
    input_size = coords.shape[1]

    model = MLP(input_size, num_classes)
    
    # --- Calculate Class Weights for Imbalance ---
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_composite = label_counts[0]
    num_prime = label_counts[1]

    # More aggressive weights for the minority class (Prime)
    # The sum of weights should typically be num_classes for CrossEntropyLoss
    # Let's give Prime a much higher weight, e.g., 20x the composite weight
    weight_composite = total_samples / (num_classes * num_composite)
    weight_prime = total_samples / (num_classes * num_prime) * 10 # Multiplied by 10 for more aggressive weighting
    
    class_weights = torch.tensor([weight_composite, weight_prime], dtype=torch.float32)
    print(f"\nCalculated Class Weights: Composite (0): {class_weights[0]:.4f}, Prime (1): {class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights) # Apply weights to loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Adjusted learning rate

    dataset = CoordinateDataset(coords, labels)

    print(f"\nLabel distribution before train/test split: {label_counts}")
    label_map_full = {0: 'Composite', 1: 'Prime'}
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

    print(f"\nTraining MLP on {len(train_dataset)} samples, validating on {len(test_dataset)} samples...")

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

    return model # Return the trained model for visualization

# --- Main Execution ---
if __name__ == "__main__":
    SPIRAL_SIZE = 1001 # Start with a moderate size for initial testing
    
    print(f"--- Starting Coordinate-Based Prediction for {SPIRAL_SIZE}x{SPIRAL_SIZE} Ulam Spiral ---")
    
    print(f"Preparing primality data for size {SPIRAL_SIZE}x{SPIRAL_SIZE} (using positional encoding)...")
    coords, labels = prepare_primality_data(SPIRAL_SIZE, encoding_dim=128)
    print(f"Total data points: {len(coords)}")
    print(f"Input feature dimension for MLP: {coords.shape[1]}")

    # Train and evaluate the MLP
    trained_model = train_and_evaluate_mlp(coords, labels, num_epochs=50, batch_size=512)

    # --- Visualization of Predicted Spiral (Optional) ---
    print("\n--- Generating Predicted Ulam Spiral Visualization ---")
    predicted_grid = np.zeros((SPIRAL_SIZE, SPIRAL_SIZE), dtype=int)
    
    # Iterate through all pixels in the grid and get model's prediction
    for r in range(SPIRAL_SIZE):
        for c in range(SPIRAL_SIZE):
            # Apply positional encoding to y and x using the SAME logic as prepare_primality_data
            max_coord_val = SPIRAL_SIZE - 1
            pe_y = get_sinusoidal_positional_encoding(r / max_coord_val * 500.0, 128)
            pe_x = get_sinusoidal_positional_encoding(c / max_coord_val * 500.0, 128)
            normalized_spiral_size = torch.tensor([SPIRAL_SIZE / 5001.0])
            
            # Prepare input tensor for model
            input_coords = torch.cat((pe_y, pe_x, normalized_spiral_size)).unsqueeze(0)
            
            # Get prediction (0 for composite, 1 for prime)
            with torch.no_grad():
                output = trained_model(input_coords)
                _, predicted_label = torch.max(output, 1)
                
            # Map back to 0 or 1 for visualization
            predicted_grid[r, c] = predicted_label.item()

    # Plotting the predicted spiral (e.g., black for composite, white for prime)
    plt.figure(figsize=(10, 10))
    plt.imshow(predicted_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest', origin='lower')
    plt.title(f"Predicted Primality Ulam Spiral ({SPIRAL_SIZE}x{SPIRAL_SIZE}) from Coordinates")
    plt.axis('off')
    plt.show()

    # Optional: Plotting the original spiral for comparison
    print("\n--- Generating Original Primality Ulam Spiral Visualization for Comparison ---")
    original_n_grid, _ = generate_ulam_spiral_and_numbers(SPIRAL_SIZE)
    original_primality_grid = np.zeros((SPIRAL_SIZE, SPIRAL_SIZE), dtype=int)
    for r in range(SPIRAL_SIZE):
        for c in range(SPIRAL_SIZE):
            n_val = original_n_grid[r, c]
            if n_val > 0:
                original_primality_grid[r, c] = 1 if is_prime(n_val) else 0
            else:
                original_primality_grid[r, c] = 0 # Or some other background for empty cells

    plt.figure(figsize=(10, 10))
    plt.imshow(original_primality_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest', origin='lower')
    plt.title(f"Original Primality Ulam Spiral ({SPIRAL_SIZE}x{SPIRAL_SIZE})")
    plt.axis('off')
    plt.show()