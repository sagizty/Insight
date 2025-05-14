import torch
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def z_sampling(image_features: torch.Tensor, coords_yx: torch.Tensor) -> tuple:
    """Reorders images and coordinates based on a Z-order scan applied tile-wise"""
    unique_x_coords = sorted(list(set(int(x.item()) for x in coords_yx[:, 0])))
    unique_y_coords = sorted(list(set(int(y.item()) for y in coords_yx[:, 1])))

    if not unique_x_coords or not unique_y_coords:  # No valid coordinates to form a grid
        return image_features, coords_yx

    H_dense = len(unique_y_coords)
    W_dense = len(unique_x_coords)
    min_hw = min(H_dense, W_dense)
    scan_level_size = 2 ** (min_hw.bit_length() - 1)

    # Create a mapping from original sparse coordinates to dense grid indices
    x_map = {x_val: i for i, x_val in enumerate(unique_x_coords)}
    y_map = {y_val: i for i, y_val in enumerate(unique_y_coords)}

    # Create a full grid storing the original indices of the images/coords, initialize with -1 to indicate empty cells in the dense grid
    full_grid = -torch.ones((H_dense, W_dense), dtype=torch.long)
    for idx, (x_sparse, y_sparse) in enumerate(coords_yx):
        # Get dense grid indices, if coordinate is recognized
        xi_dense = x_map.get(int(x_sparse.item()))
        yi_dense = y_map.get(int(y_sparse.item()))

        if xi_dense is not None and yi_dense is not None:
            full_grid[yi_dense, xi_dense] = idx

    def generate_z_order_path(size: int) -> list[tuple[int, int]]:
        if not (size > 0 and (size & (size - 1) == 0)):
            raise ValueError(f"Size must be a power of 2 and greater than 0. Got {size}")

        if size == 1:
            return [(0, 0)]

        # Recursively get the path for a quadrant
        sub_size = size // 2
        sub_path = generate_z_order_path(sub_size)

        path = []
        # Quadrant 0 (Top-Left)
        for sx, sy in sub_path:
            path.append((sx, sy))
        # Quadrant 1 (Top-Right)
        for sx, sy in sub_path:
            path.append((sx + sub_size, sy))
        # Quadrant 2 (Bottom-Left)
        for sx, sy in sub_path:
            path.append((sx, sy + sub_size))
        # Quadrant 3 (Bottom-Right)
        for sx, sy in sub_path:
            path.append((sx + sub_size, sy + sub_size))
        return path

    # Generate the Z-order path for the specified tile size
    try:
        z_path_pattern = generate_z_order_path(scan_level_size)
    except ValueError as e:
        print(f"Error generating Z-path: {e}")
        print(f"Defaulting to original order or consider providing a valid power-of-2 scan_level_size.")
        return image_features, coords_yx

    reordered_original_indices = []

    # Iterate over the dense grid in blocks of size tile_h x tile_w
    for block_y_start_dense in range(0, H_dense, scan_level_size):
        for block_x_start_dense in range(0, W_dense, scan_level_size):
            # Apply the Z-path within this block
            for dx_in_tile, dy_in_tile in z_path_pattern:
                # Calculate actual dense grid coordinates
                current_dense_x = block_x_start_dense + dx_in_tile
                current_dense_y = block_y_start_dense + dy_in_tile

                # Check if these coordinates are within the bounds of our dense grid
                if 0 <= current_dense_x < W_dense and 0 <= current_dense_y < H_dense:
                    original_idx = full_grid[current_dense_y, current_dense_x]
                    if original_idx != -1:  # If there's an image patch at this location
                        reordered_original_indices.append(original_idx.item())

    reordered_indices_tensor = torch.tensor(reordered_original_indices, dtype=torch.long, device=image_features.device)
    return image_features[reordered_indices_tensor], coords_yx[reordered_indices_tensor]


def local_box_sampling(image_features, coords_yx, bag_num=60, bag_size=64,
                       neighbourhood_radius=4, tile_pixel_size=224, max_ittr=5):
    """
    Args:
        image_features: [N, C] tensor of tile features
        coords_yx: [N, 2] array-like, coordinates in pixels (Y, X)
        bag_num: number of bags to sample
        bag_size: number of tiles per bag
        neighbourhood_radius: radius in tile units
        tile_pixel_size: size of tile in pixels
        max_ittr: how many times to attempt center selection

    Returns:
        sampled_features: [bag_num * bag_size, C] tensor
        sampled_coords: [bag_num * bag_size, 2] array of coordinates
    """

    coords_yx = np.array(coords_yx)
    N = coords_yx.shape[0]
    all_indices = np.arange(N)

    # Build KDTree for fast neighbor search
    tree = KDTree(coords_yx)

    sampled_indices = []
    attempts = 0
    bag_valid_flag = 0

    shuffled_indices = np.random.permutation(N)

    max_distance = neighbourhood_radius * tile_pixel_size

    for center_idx in shuffled_indices:
        center_coord = coords_yx[center_idx]
        neighbor_indices = tree.query_ball_point(center_coord, r=max_distance, p=1)  # Manhattan distance

        if len(neighbor_indices) < bag_size:
            continue

        selected = np.random.choice(neighbor_indices, size=bag_size, replace=False)
        sampled_indices.extend(selected.tolist())
        bag_valid_flag += 1

        if bag_valid_flag == bag_num:
            break

        attempts += 1
        if attempts > max_ittr * N:
            print("Warning: insufficient valid bags, returning partial result")
            break

    if bag_valid_flag < bag_num:
        return -1

    sampled_features = image_features[sampled_indices]  # [bag_num * bag_size, C]
    sampled_coords = coords_yx[sampled_indices]  # [bag_num * bag_size, 2]

    return sampled_features, sampled_coords


def visualize_sampled_boxes(coords_yx, sampled_coords, bag_size=64):
    """
    Visualize sampled local boxes over the full coordinate set.

    Args:
        coords_yx: [N, 2] full tile coordinates (Y, X)
        sampled_coords: [bag_num * bag_size, 2] sampled coordinates
        bag_size: number of tiles per bag
    """
    coords_yx = np.array(coords_yx)
    sampled_coords = np.array(sampled_coords)

    # Plot all tiles in light gray
    plt.figure(figsize=(8, 8))
    plt.scatter(coords_yx[:, 1], coords_yx[:, 0], s=10, c='lightgray', label='All Tiles')

    # Color each bag separately
    bag_num = len(sampled_coords) // bag_size
    cmap = plt.get_cmap('tab20')  # up to 20 distinct colors

    for i in range(bag_num):
        bag = sampled_coords[i * bag_size: (i + 1) * bag_size]
        plt.scatter(bag[:, 1], bag[:, 0], s=15, label=f'Bag {i+1}', color=cmap(i % 20))

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title('Sampled Local Bags on Tile Coordinates')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend(fontsize='small', ncol=2, loc='upper right')
    plt.tight_layout()
    plt.show()


def visualize_z_order_path(coords_yx: torch.Tensor):
    """
    Visualize the tile coordinates following the Z-order (Morton order) traversal.

    Args:
        coords_yx: Tensor of shape [N, 2] where each row is [y, x] in pixel coordinates
                   (should already be reordered by z_sampling)
    """
    coords_yx_np = coords_yx.cpu().numpy() if torch.is_tensor(coords_yx) else np.array(coords_yx)

    plt.figure(figsize=(8, 8))
    for i, (y, x) in enumerate(coords_yx_np):
        plt.scatter(x, y, c='blue', s=10)
        plt.text(x, y, str(i), fontsize=6, color='red')  # index as order label

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title("Z-order Sampling Path (Tile Traversal Order)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Simulate 1 WSI
    num_tiles = 2000
    feature_dim = 768
    max_coord = 22400  # WSI size in pixels

    coords_yx = np.random.randint(0, max_coord, size=(num_tiles, 2))  # [N, 2]
    image_features = torch.randn(num_tiles, feature_dim)  # [N, C]

    sampled_features, sampled_coords = local_box_sampling(image_features, coords_yx,
                                bag_num=10, bag_size=20,  # smaller for visibility
                                neighbourhood_radius=10,
                                tile_pixel_size=224)
    visualize_sampled_boxes(coords_yx, sampled_coords, bag_size=20)

    # Apply Z-order sampling
    reordered_features, reordered_coords = z_sampling(image_features, coords_yx)
    # Visualize the tile order
    visualize_z_order_path(reordered_coords)






