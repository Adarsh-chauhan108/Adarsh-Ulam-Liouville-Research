import numpy as np
import matplotlib.pyplot as plt

def liouville_lambda(n):
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

def generate_ulam_spiral(size):
    grid = np.zeros((size, size), dtype=int)
    x, y = size // 2, size // 2
    dx, dy = 0, -1
    n = 1
    for _ in range(size * size):
        if 0 <= x < size and 0 <= y < size:
            grid[y, x] = liouville_lambda(n)
        if x == y or (x < y and x + y == size - 1) or (x > y and x + y == size):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
        n += 1
    return grid

def analyze_corner_tip(grid, zoom=5):
    size = grid.shape[0]
    # Top-right corner is at [0, size-1]
    y, x = 0, size - 1

    # Determine bounds of zoom window (e.g., 5x5 centered around corner)
    half = zoom // 2
    ymin = max(0, y - half)
    ymax = min(size, y + half + 1)
    xmin = max(0, x - half)
    xmax = min(size, x + half + 1)

    region = grid[ymin:ymax, xmin:xmax]
    red_count = np.sum(region == -1)
    blue_count = np.sum(region == 1)

    print(f"🎯 Top-Right Corner Tip Zoom ({zoom}x{zoom}):")
    print(f"Red (λ=-1): {red_count}")
    print(f"Blue (λ=+1): {blue_count}")
    print(f"Red %: {(red_count / region.size) * 100:.2f}%")
    print(f"Values around corner tip:\n{region}")

    # Visualize
    plt.figure(figsize=(4, 4))
    plt.imshow(region, cmap='bwr', interpolation='nearest')
    plt.title(f"Top-Right Corner Tip ({zoom}x{zoom})")
    plt.axis('off')
    plt.show()

# Test
size = 3001  # You can try 1301, 1201, etc.
grid = generate_ulam_spiral(size)
analyze_corner_tip(grid, zoom=7)  # Zoom in on 7x7 tip area