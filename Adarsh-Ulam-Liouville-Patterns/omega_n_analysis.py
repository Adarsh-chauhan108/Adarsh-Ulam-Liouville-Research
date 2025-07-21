import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: Liouville Function and Ulam Spiral Generation (Your Code) ---

def liouville_lambda(n):
    """Calculates the Liouville function lambda(n)."""
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
    return 1 if omega % 2 == 0 else -1 # 1 for +1, -1 for -1

def generate_ulam_spiral_and_numbers(size):
    """
    Generates an Ulam spiral grid with Liouville lambda(n) values
    and a separate grid with the actual n values.
    """
    lambda_grid = np.zeros((size, size), dtype=int)
    n_grid = np.zeros((size, size), dtype=int) # To store actual n values

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

    # Populate both grids
    for r in range(size):
        for c in range(size):
            if (r, c) in coords_to_n:
                current_n = coords_to_n[(r, c)]
                n_grid[r, c] = current_n
                lambda_grid[r, c] = liouville_lambda(current_n)
            else:
                n_grid[r, c] = 0 # Mark as empty or outside spiral bounds
                lambda_grid[r, c] = 0 # Neutral value for empty spots

    return lambda_grid, n_grid

# --- Function to Extract Red N-values from a Region ---

def extract_red_n_values_from_region(lambda_grid, n_grid, region_coords):
    """
    Extracts the actual 'n' values from a specified region where lambda(n) is -1 (red).

    Args:
        lambda_grid (np.ndarray): Grid with lambda(n) values (-1 or 1).
        n_grid (np.ndarray): Grid with actual integer 'n' values.
        region_coords (tuple): (ymin, ymax, xmin, xmax) defining the rectangular region.

    Returns:
        list: A list of 'n' values where lambda(n) is -1 within the region.
    """
    ymin, ymax, xmin, xmax = region_coords
    red_n_values = []

    for r in range(ymin, ymax):
        for c in range(xmin, xmax):
            if 0 <= r < lambda_grid.shape[0] and 0 <= c < lambda_grid.shape[1]: # Ensure within grid bounds
                if lambda_grid[r, c] == -1: # If it's a red pixel
                    red_n_values.append(n_grid[r, c])
    return red_n_values

# --- Helper Function for Prime Factorization (Needed for Omega(n)) ---
def count_prime_factors_with_multiplicity(num):
    """Counts total prime factors (Omega(n)) for a number."""
    if num <= 1:
        return 0
    count = 0
    d = 2
    temp_num = num
    while d * d <= temp_num:
        while temp_num % d == 0:
            count += 1
            temp_num //= d
        d += 1
    if temp_num > 1: # Remaining number is a prime factor
        count += 1
    return count

# --- Main Execution for Characterization ---
if __name__ == "__main__":
    SPIRAL_SIZE = 5001 # Start with a size where you clearly saw the L
    # Define the top-right corner region where you observe the 'Reverse L'
    # These coordinates might need fine-tuning based on your visual inspection
    # This is a 15% x 15% square from the top-right
    corner_percentage = 0.15
    corner_pixels = int(SPIRAL_SIZE * corner_percentage)

    # ymin, ymax, xmin, xmax for the region (remember y=0 is top row)
    region_ymin = 0
    region_ymax = corner_pixels
    region_xmin = SPIRAL_SIZE - corner_pixels
    region_xmax = SPIRAL_SIZE

    print(f"Generating Ulam spiral for size {SPIRAL_SIZE}x{SPIRAL_SIZE}...")
    lambda_grid, n_grid = generate_ulam_spiral_and_numbers(SPIRAL_SIZE)
    print("Ulam spiral generated.")

    print(f"Extracting 'n' values for red pixels in the top-right {corner_percentage*100:.0f}% corner region...")
    red_n_values = extract_red_n_values_from_region(lambda_grid, n_grid,
                                                    (region_ymin, region_ymax, region_xmin, region_xmax))

    print(f"\nFound {len(red_n_values)} red 'n' values in the specified region.")

    if red_n_values:
        print("\nAnalyzing properties of the first 20 red 'n' values:")
        for i, n_val in enumerate(red_n_values[:20]):
            omega_n = count_prime_factors_with_multiplicity(n_val)
            print(f"n = {n_val:<8}, Omega(n) = {omega_n:<3}, λ(n) = {liouville_lambda(n_val)}")
            if liouville_lambda(n_val) != -1:
                print(f"  WARNING: Expected λ(n)=-1 for n={n_val}, but got {liouville_lambda(n_val)}. Check region definition or lambda calculation.")

        print("\nCommon properties to look for (manually inspect the list):")
        print("- Are they mostly odd or even?")
        print("- Do they have a common factor (e.g., all multiples of 3, 5, 7, etc.)?")
        print("- Do they seem to follow a polynomial sequence (e.g., n^2 + an + b)?")
        print("- Are their Omega(n) values consistently odd (as expected for λ(n)=-1)?")
        print("- Are there any perfect squares or near-squares among them? (λ(square) is always +1, so these would be exceptions if found in a red region)")

    else:
        print("No red values found in the specified region. Adjust region_coords or spiral_size.")