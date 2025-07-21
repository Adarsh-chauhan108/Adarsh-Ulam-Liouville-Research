import numpy as np
import matplotlib.pyplot as plt

def liouville_lambda(n):
    if n < 1: return 0
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
    lambda_grid = np.zeros((size, size), dtype=int)
    n_grid = np.zeros((size, size), dtype=int)
    n_to_coords = {} # New: Map n to (y, x)

    x, y = size // 2, size // 2
    dx, dy = 0, -1
    n = 1
    steps_taken = 0
    segment_length = 1
    turns_in_segment = 0

    while n <= size * size:
        if 0 <= y < size and 0 <= x < size:
            n_grid[y, x] = n
            lambda_grid[y, x] = liouville_lambda(n)
            n_to_coords[n] = (y, x) # Store the coordinate for n
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

    return lambda_grid, n_grid, n_to_coords # Return the new dictionary

def extract_red_n_values_from_region(lambda_grid, n_grid, region_coords):
    ymin, ymax, xmin, xmax = region_coords
    red_n_values = []

    for r in range(ymin, ymax):
        for c in range(xmin, xmax):
            if 0 <= r < lambda_grid.shape[0] and 0 <= c < lambda_grid.shape[1]:
                if lambda_grid[r, c] == -1:
                    red_n_values.append(n_grid[r, c])
    return red_n_values

def count_prime_factors_with_multiplicity(num):
    if num <= 1: return 0
    count = 0
    d = 2
    temp_num = num
    while d * d <= temp_num:
        while temp_num % d == 0:
            count += 1
            temp_num //= d
        d += 1
    if temp_num > 1:
        count += 1
    return count

# Main Execution for Characterization
if __name__ == "__main__":
    SPIRAL_SIZE = 501 # Start with a size where you clearly saw the L
    corner_percentage = 0.15
    corner_pixels = int(SPIRAL_SIZE * corner_percentage)

    region_ymin = 0
    region_ymax = corner_pixels
    region_xmin = SPIRAL_SIZE - corner_pixels
    region_xmax = SPIRAL_SIZE

    print(f"Generating Ulam spiral for size {SPIRAL_SIZE}x{SPIRAL_SIZE}...")
    lambda_grid, n_grid, n_to_coords = generate_ulam_spiral_and_numbers(SPIRAL_SIZE) # Get n_to_coords
    print("Ulam spiral generated.")

    print(f"Extracting 'n' values for red pixels in the top-right {corner_percentage*100:.0f}% corner region...")
    red_n_values = extract_red_n_values_from_region(lambda_grid, n_grid,
                                                    (region_ymin, region_ymax, region_xmin, region_xmax))

    print(f"\nFound {len(red_n_values)} red 'n' values in the specified region.")

    # --- New: Extract and Analyze Coordinates of Red Pixels ---
    red_coords = []
    for n_val in red_n_values:
        if n_val in n_to_coords: # Ensure n_val is valid (not 0 from padding)
            red_coords.append(n_to_coords[n_val])

    print(f"\nAnalyzing coordinates of {len(red_coords)} red pixels:")
    if red_coords:
        # Plotting the coordinates to visualize the 'L'
        y_coords = [coord[0] for coord in red_coords]
        x_coords = [coord[1] for coord in red_coords]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, s=1, color='red', alpha=0.5) # s=1 for small dots
        plt.title(f"Red Pixels in Top-Right Corner (Size {SPIRAL_SIZE})")
        plt.gca().set_aspect('equal', adjustable='box') # Make sure it's square
        plt.xlim(region_xmin, region_xmax) # Zoom into the region
        plt.ylim(region_ymin, region_ymax) # Zoom into the region
        plt.gca().invert_yaxis() # Invert y-axis to match image convention (y=0 at top)
        plt.show()

        # Analyze coordinate patterns (first 20 for brevity)
        print("\nFirst 20 red pixel coordinates (y, x):")
        for i, coord in enumerate(red_coords[:20]):
            print(f"({coord[0]:<4}, {coord[1]:<4})")

        # Look for polynomial relationships in coordinates
        # This is a manual inspection step initially
        print("\nLook for patterns in coordinates:")
        print("- Do they form straight lines? What are their slopes?")
        print("- Do they form curves? Can you fit a polynomial (y = ax^2 + bx + c or x = ay^2 + by + c)?")
        print("- What is their distance from the top edge (y) and right edge (x_max - x)?")

    else:
        print("No red coordinates found.")

    # (Optional: Keep the n-value analysis from before if you want to re-run it)
    # print("\nAnalyzing properties of the first 20 red 'n' values:")
    # for i, n_val in enumerate(red_n_values[:20]):
    #     omega_n = count_prime_factors_with_multiplicity(n_val)
    #     print(f"n = {n_val:<8}, Omega(n) = {omega_n:<3}, λ(n) = {liouville_lambda(n_val)}")