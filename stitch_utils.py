import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from pathlib import Path
import seaborn as sns

import pathlib
import boto3

import io_utils


def analyze_tile_grid(tile_dict, plot=True):
    """
    Analyze the tile grid structure and show coverage with visualization
    Args:
        tile_dict: Dictionary of tile names key: tile_id, value: tile_name
        Example. {0: 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr', 1: 'Tile_X_0001_Y_0000_Z_0000_ch_405.zarr'}
        plot: Whether to show the coverage plot
    Returns:
        dict: Detailed information about the tile grid
    """
    # Extract coordinates
    coords = []
    for _, tile in tile_dict.items():
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        x = int(parts[2])
        y = int(parts[4])
        z = int(parts[6])
        coords.append((x, y, z))
    
    # Find dimensions
    x_coords = {x for x, _, _ in coords}
    y_coords = {y for _, y, _ in coords}
    z_coords = {z for _, _, z in coords}
    
    x_dim = max(x_coords) + 1
    y_dim = max(y_coords) + 1
    z_dim = max(z_coords) + 1
    
    # Create coverage map
    coverage = np.zeros((y_dim, x_dim))  # Note: y_dim first for correct plotting
    for x, y, _ in coords:
        coverage[y, x] = 1
    
    if plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(coverage, cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(label='Tile Present')
        plt.title(f'Tile Coverage Map ({len(tile_dict)} tiles)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Add grid lines
        plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add coordinate labels
        for i in range(x_dim):
            for j in range(y_dim):
                color = 'white' if coverage[j, i] == 0 else 'black'
                plt.text(i, j, f'({i},{j})', ha='center', va='center', color=color)
        
        plt.show()
    
    # Calculate statistics
    theoretical_tiles = x_dim * y_dim * z_dim
    actual_tiles = len(tile_dict)
    
    # Find missing coordinates
    all_coords = {(x, y, z) for x in range(x_dim) 
                           for y in range(y_dim) 
                           for z in range(z_dim)}
    present_coords = set(coords)
    missing_coords = all_coords - present_coords
    
    return {
        'dimensions': (x_dim, y_dim, z_dim),
        'theoretical_tiles': theoretical_tiles,
        'actual_tiles': actual_tiles,
        'coverage_percentage': (actual_tiles / theoretical_tiles) * 100,
        'x_range': (min(x_coords), max(x_coords)),
        'y_range': (min(y_coords), max(y_coords)),
        'z_range': (min(z_coords), max(z_coords)),
        'missing_coords': sorted(missing_coords),
        'present_coords': sorted(present_coords),
        'coverage_map': coverage
    }



def plot_tile_transforms(tile_dict, transforms, coverage_map):
    """
    Plot tiles with arrows showing their transformation vectors
    Args:
        tile_dict: Dictionary mapping IDs to tile names
        transforms: defaultdict of transformation matrices as numpy arrays
        coverage_map: 2D numpy array showing tile presence
    """
    y_dim, x_dim = coverage_map.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the tile coverage base
    im = ax.imshow(coverage_map, cmap='RdYlGn', interpolation='nearest', alpha=0.3)
    
    # Keep track of magnitudes for color scaling
    magnitudes = []
    
    # First pass to get magnitude range
    for tile_id, tile_name in tile_dict.items():
        transform = transforms[tile_id]
        dx = transform[0, 3]
        dy = transform[1, 3]
        magnitudes.append(np.sqrt(dx**2 + dy**2))
    
    max_magnitude = max(magnitudes)
    
    # Plot transformation vectors for each tile
    for tile_id, tile_name in tile_dict.items():
        # Get tile coordinates
        parts = tile_name.split('_ch_')[0].split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        # Get transformation
        transform = transforms[tile_id]
        dx = transform[0, 3]
        dy = transform[1, 3]
        
        # Calculate magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Color based on magnitude
        color = plt.cm.viridis(magnitude / max_magnitude)
        
        # Scale factor for visualization (adjust as needed)
        scale = 1/10000  # This might need adjustment based on your transform magnitudes
        
        # Plot arrow from tile center
        ax.arrow(tile_x, tile_y,           # Start at tile position
                dx * scale, dy * scale,     # Scaled displacement
                head_width=0.1,
                head_length=0.1,
                fc=color, ec=color,
                alpha=0.7)
        
        # Add magnitude text
        ax.text(tile_x, tile_y, f'{magnitude:.0f}', 
               ha='center', va='bottom', color='black')
    
    ax.set_title('Tile Transformation Vectors')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    # Add scale reference
    scale_length = 1000 * scale  # Length in plot units
    ax.arrow(x_dim-1, y_dim-1, scale_length, 0,
            head_width=0.1, head_length=0.1,
            fc='red', ec='red',
            label='Scale')
    ax.text(x_dim-1, y_dim-1.3, f'1000 pixels', ha='center')
    
    # Add colorbar
    norm = plt.Normalize(vmin=0, vmax=max_magnitude)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Transform magnitude (pixels)')
    
    plt.tight_layout()
    plt.show()



def plot_transform_heatmaps(tile_dict, 
                            transforms, 
                            coverage_map, 
                            remove_nominal_transform=False, 
                            tile_dims=None,
                            overlap=0.0):
    """
    Plot three heatmaps showing X, Y, and Z transformations with colormaps centered on zero
    Args:
        tile_dict: Dictionary mapping IDs to tile names
        transforms: defaultdict of transformation matrices as numpy arrays
        coverage_map: 2D numpy array showing tile presence
        remove_nominal_transform: Whether to remove the nominal transform from the heatmaps

    The nominal transform is the transform that maps the tile to the nominal coordinate system
    tile_dims: tuple of (x_dim, y_dim, z_dim) representing the dimensions of tile in pixels
    We use the tile_dims to remove the nominal transform from the heatmaps. 
    For X transforms, we remove the x_dim from the transform. We need to get the X coordinate index, starting from the middle.
    So if we have 7 tiles in the X direction, the middle tile is index 3. The scale index vector in [-3,-2,-1,0,1,2,3] * tile_dims[0]
   
   For Y transforms, we remove the y_dim from the transform. 
    """
    y_dim, x_dim = coverage_map.shape
    
    # assert that tile_dims is not None if remove_nominal_transform is True
    if remove_nominal_transform and tile_dims is None:
        raise ValueError("tile_dims must be provided if remove_nominal_transform is True")
    
    # Create arrays to store the transform values
    x_transforms = np.full_like(coverage_map, np.nan, dtype=float)
    y_transforms = np.full_like(coverage_map, np.nan, dtype=float)
    z_transforms = np.full_like(coverage_map, np.nan, dtype=float)

    x_scale_index = np.arange(x_dim) - (x_dim - 1) / 2
    y_scale_index = np.arange(y_dim) - (y_dim - 1) / 2
    x_scale_index = x_scale_index * tile_dims[0] * (1 - overlap)
    y_scale_index = y_scale_index * tile_dims[1] * (1 - overlap)
    print(x_scale_index)
    print(y_scale_index)
    
    # Fill in the transform values
    for tile_id, tile_name in tile_dict.items():
        parts = tile_name.split('_ch_')[0].split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        transform = transforms[tile_id]
        x_transforms[tile_y, tile_x] = transform[0, 3]
        y_transforms[tile_y, tile_x] = transform[1, 3]
        z_transforms[tile_y, tile_x] = transform[2, 3]

        if remove_nominal_transform:
            x_transforms[tile_y, tile_x] = x_transforms[tile_y, tile_x] - x_scale_index[tile_x]
            y_transforms[tile_y, tile_x] = y_transforms[tile_y, tile_x] - y_scale_index[tile_y]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Function to get symmetric vmin/vmax for centered colormap
    def get_symmetric_limits(data):
        valid_data = data[~np.isnan(data)]
        abs_max = np.max(np.abs(valid_data))
        return -abs_max, abs_max
    
    # Plot X transforms
    vmin_x, vmax_x = get_symmetric_limits(x_transforms)
    im1 = ax1.imshow(x_transforms, cmap='RdBu', vmin=vmin_x, vmax=vmax_x)
    ax1.set_title('X Transforms')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    fig.colorbar(im1, ax=ax1, label='X translation (pixels)')
    
    # Plot Y transforms
    vmin_y, vmax_y = get_symmetric_limits(y_transforms)
    im2 = ax2.imshow(y_transforms, cmap='RdBu', vmin=vmin_y, vmax=vmax_y)
    ax2.set_title('Y Transforms')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    fig.colorbar(im2, ax=ax2, label='Y translation (pixels)')
    
    # Plot Z transforms
    vmin_z, vmax_z = get_symmetric_limits(z_transforms)
    im3 = ax3.imshow(z_transforms, cmap='RdBu', vmin=vmin_z, vmax=vmax_z)
    ax3.set_title('Z Transforms')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    fig.colorbar(im3, ax=ax3, label='Z translation (pixels)')
    
    # Add grid lines
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_transforms, y_transforms, z_transforms


from collections import defaultdict
import numpy as np

def calculate_net_transforms(
    view_transforms: dict[int, list[dict]]
) -> dict[int, np.ndarray]:
    """
    Accumulate net transform and net translation for each matrix stack.
    Net translation =
        Sum of translation vectors converted into original nominal basis
    Net transform =
        Product of 3x3 matrices
    NOTE: Translational component (last column) is defined
          wrt to the DOMAIN, not codomain.
          Implementation is informed by this given.

    NOTE: Carsons version 2/21
    Parameters
    ------------------------
    view_transforms: dict[int, list[dict]]
        Dictionary of tile ids to transforms associated with each tile.

    Returns
    ------------------------
    dict[int, np.ndarray]:
        Dictionary of tile ids to net transform.

    """

    identity_transform = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    net_transforms: dict[int, np.ndarray] = defaultdict(
        lambda: np.copy(identity_transform)
    )

    for view, tfs in view_transforms.items():
        net_translation = np.zeros(3)
        net_matrix_3x3 = np.eye(3)
        curr_inverse = np.eye(3)

        for (tf) in (tfs):  # Tfs is a list of dicts containing transform under 'affine' key
            nums = [float(val) for val in tf["affine"].split(" ")]
            matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
            translation = np.array(nums[3::4])
            
            # print(translation)
            nums = np.array(nums).reshape(3,4)
            matrix_3x3 = np.array([nums[:,0], nums[:,1], nums[:,2]]).T
            translation = np.array(nums[:,3])
            
            #old way
            net_translation = net_translation + (curr_inverse @ translation)
            net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3  
            curr_inverse = np.linalg.inv(net_matrix_3x3)  # Update curr_inverse

            #new way
            #net_translation = net_translation + (translation)
            #net_matrix_3x3 = net_matrix_3x3 @ matrix_3x3 

        net_transforms[view] = np.hstack(
            (net_matrix_3x3, net_translation.reshape(3, 1))
        )

    return net_transforms


def get_tile_grid_dimensions(tile_names):
    """
    Determine the dimensions of the tile grid (max X, Y, Z coordinates)
    Args:
        tile_names: List of tile names in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
    Returns:
        tuple: (max_x + 1, max_y + 1, max_z + 1) representing grid dimensions
    """
    x_coords = set()
    y_coords = set()
    z_coords = set()
    
    for tile in tile_names:
        # Split by '_ch_' to remove suffix and then split remaining parts
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        
        x_coords.add(int(parts[2]))  # X coordinate
        y_coords.add(int(parts[4]))  # Y coordinate
        z_coords.add(int(parts[6]))  # Z coordinate
    
    # Add 1 to each max coordinate to get dimensions
    dimensions = (
        max(x_coords) + 1,
        max(y_coords) + 1,
        max(z_coords) + 1
    )
    
    return dimensions


def parse_tile_name(tile_name):
    """Extract X, Y, Z coordinates from a tile name like 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'"""
    # Remove the suffix and split by underscore
    base_name = get_base_tile_name(tile_name)
    parts = base_name.split('_')
    x = int(parts[2])
    y = int(parts[4])
    z = int(parts[6])
    ch = int(tile_name.split('_ch_')[1].split('.zarr')[0])
    return (x, y, z), ch

def get_base_tile_name(tile_name):
    """Get the base tile name without the channel and extension"""
    return tile_name.split('_ch_')[0]


def get_adjacent_tiles(tile_name, existing_tiles, include_diagonals=True):
    """
    Get the names of adjacent tiles that exist in the provided list
    Args:
        tile_name: Name of the tile in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
        existing_tiles: List of valid tile names to check against
        include_diagonals: If True, includes diagonal neighbors
                          If False, only cardinal directions
    """
    (x, y, z), ch = parse_tile_name(tile_name)
    
    # Convert existing_tiles to base names for comparison
    existing_base_tiles = {get_base_tile_name(t) for t in existing_tiles}
    
    adjacent_tiles = []
    
    # Choose which directions to check
    if include_diagonals:
        directions = [(dx, dy) for dx in [-1, 0, 1] 
                             for dy in [-1, 0, 1] 
                             if not (dx == 0 and dy == 0)]
    else:
        directions = [(0, 1),   # North
                     (1, 0),    # East
                     (0, -1),   # South
                     (-1, 0)]   # West
    
    # Generate and filter adjacent tile names
    for dx, dy in directions:
        adj_x = str(x + dx).zfill(4)
        adj_y = str(y + dy).zfill(4)
        adj_z = str(z).zfill(4)
        
        adjacent_base_name = f"Tile_X_{adj_x}_Y_{adj_y}_Z_{adj_z}"
        if adjacent_base_name in existing_base_tiles:
            # Find the full tile name from the original list
            full_name = [t for t in existing_tiles 
                        if get_base_tile_name(t) == adjacent_base_name][0]
            adjacent_tiles.append(full_name)
    
    return adjacent_tiles

def get_all_adjacent_pairs(tile_names, include_diagonals=False):
    """
    Generate all pairs of adjacent tiles in the dataset
    
    Args:
        tile_names: List of tile names in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
        include_diagonals: If True, includes diagonal neighbors
                          If False, only cardinal directions
    
    Returns:
        list: List of tuples containing pairs of adjacent tile names
              Each pair is ordered (tile1, tile2) where tile1 has a lower 
              index in the original tile_names list than tile2
    """
    pairs = []
    
    # Convert to list if dictionary is provided
    if isinstance(tile_names, dict):
        tile_names = list(tile_names.values())
    
    # For each tile, find its adjacent tiles
    for i, tile in enumerate(tile_names):
        adjacent_tiles = get_adjacent_tiles(tile, tile_names, include_diagonals)
        
        # Only keep pairs where the adjacent tile has a higher index
        # This prevents duplicate pairs and ensures consistent ordering
        for adj_tile in adjacent_tiles:
            j = tile_names.index(adj_tile)
            if i < j:  # Only add if current tile index is lower
                pairs.append((tile, adj_tile))
    
    return pairs

def analyze_adjacent_pairs(pairs, tile_names, transforms):
    """
    Analyze the transformation differences between adjacent tile pairs
    
    Args:
        pairs: List of tuples containing pairs of adjacent tile names
        transforms: Dictionary mapping tile names or IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
    
    Returns:
        dict: Statistics about the transformations between adjacent tiles
              Including mean, std, min, max of translation differences
    """
    # Store differences for each dimension
    x_diffs = []
    y_diffs = []
    z_diffs = []
    
    for tile1, tile2 in pairs:
        # Get transforms for both tiles
        if isinstance(transforms, dict):
            # get index of tile1 and tile2 in tile_names
            idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1)]
            idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2)]
            t1 = transforms[idx1]
            t2 = transforms[idx2]
        
        # Calculate differences in translation components
        x_diff = abs(t1[0, 3] - t2[0, 3])
        y_diff = abs(t1[1, 3] - t2[1, 3])
        z_diff = abs(t1[2, 3] - t2[2, 3])
        
        x_diffs.append(x_diff)
        y_diffs.append(y_diff)
        z_diffs.append(z_diff)
    
    # Calculate statistics
    stats = {
        'x_translation': {
            'mean': np.mean(x_diffs),
            'std': np.std(x_diffs),
            'min': np.min(x_diffs),
            'max': np.max(x_diffs),
            'x_diffs': x_diffs
        },
        'y_translation': {
            'mean': np.mean(y_diffs),
            'std': np.std(y_diffs),
            'min': np.min(y_diffs),
            'max': np.max(y_diffs),
            'y_diffs': y_diffs
        },
        'z_translation': {
            'mean': np.mean(z_diffs),
            'std': np.std(z_diffs),
            'min': np.min(z_diffs),
            'max': np.max(z_diffs),
            'z_diffs': z_diffs
        },
        'n_pairs': len(pairs)
    }
    
    return stats


def extract_x_y_z_transforms(transforms):
    """
    Extract x, y, z transforms from a dictionary of transforms
    """
    transform = list(transforms.values())
    x_transforms = [t[0, 3] for t in transform]
    y_transforms = [t[1, 3] for t in transform]
    z_transforms = [t[2, 3] for t in transform]

    return x_transforms, y_transforms, z_transforms


def analyze_outlier_pairs(pairs, tile_names, transforms, threshold=None):
    """
    Analyze pairs of tiles with unusually large transformation differences
    
    Args:
        pairs: List of tuples containing pairs of adjacent tile names
        tile_names: Dictionary mapping tile IDs to tile names
        transforms: Dictionary mapping tile IDs to transformation matrices
        threshold: Optional float to define outlier threshold. If None,
                  uses mean + 2*std
    
    Returns:
        dict: Information about outlier pairs including:
              - The tile pairs
              - Their locations
              - The magnitude of their differences
    """
    # Get basic stats first
    stats = analyze_adjacent_pairs(pairs, tile_names, transforms)
    
    # Analyze each dimension separately
    outliers = {
        'x': [],
        'y': [],
        'z': []
    }
    
    # If threshold not provided, use statistical threshold
    if threshold is None:
        thresholds = {
            'x': stats['x_translation']['mean'] + 1 * stats['x_translation']['std'],
            'y': stats['y_translation']['mean'] + 1 * stats['y_translation']['std'],
            'z': stats['z_translation']['mean'] + 1 * stats['z_translation']['std']
        }
    else:
        thresholds = {'x': threshold, 'y': threshold, 'z': threshold}
    print(thresholds)
    
    for i, (tile1, tile2) in enumerate(pairs):
        # Get indices and transforms
        idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1)]
        idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2)]
        t1 = transforms[idx1]
        t2 = transforms[idx2]
        
        # Calculate differences
        diffs = {
            'x': abs(t1[0, 3] - t2[0, 3]),
            'y': abs(t1[1, 3] - t2[1, 3]),
            'z': abs(t1[2, 3] - t2[2, 3])
        }
        
        # Get tile positions
        pos1 = parse_tile_name(tile1)
        pos2 = parse_tile_name(tile2)
        
        # Check each dimension for outliers
        for dim in ['x', 'y', 'z']:
            if diffs[dim] > thresholds[dim]:
                outliers[dim].append({
                    'tile1': {
                        'name': tile1,
                        'position': pos1,
                        'transform': t1
                    },
                    'tile2': {
                        'name': tile2,
                        'position': pos2,
                        'transform': t2
                    },
                    'difference': diffs[dim]
                })
    
    # Sort outliers by difference magnitude
    for dim in outliers:
        outliers[dim].sort(key=lambda x: x['difference'], reverse=True)
    
    return outliers


def plot_transform_differences_histogram(stats, bins=50):

    """
    Plot histograms of transform differences with additional analysis
    
    Args:
        stats: Statistics dictionary from analyze_adjacent_pairs
        bins: Number of bins for histogram
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot X differences
    ax1.hist(stats['x_translation']['x_diffs'], bins=bins)
    ax1.set_title('X Translation Differences')
    ax1.set_xlabel('Difference (pixels)')
    ax1.set_ylabel('Count')
    
    # Plot Y differences
    ax2.hist(stats['y_translation']['y_diffs'], bins=bins)
    ax2.set_title('Y Translation Differences')
    ax2.set_xlabel('Difference (pixels)')
    
    # Plot Z differences
    ax3.hist(stats['z_translation']['z_diffs'], bins=bins)
    ax3.set_title('Z Translation Differences')
    ax3.set_xlabel('Difference (pixels)')

    # # XLIMIT
    # ax1.set_xlim(0, 100)
    # ax2.set_xlim(0, 100)
    # ax3.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()


def get_transformed_tile_pair(tile1_name, tile2_name, transforms, tile_names, 
                              bucket_name, dataset_path, 
                             z_slice=None, pyramid_level=0):
    """
    Get a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Z-slice to display
        pyramid_level: Pyramid level to load
        
    Returns:
        combined: Combined image array
        extent: [x_min, x_max, y_min, y_max] for plotting
        z_slice: The z-slice that was used
    """
    # Get indices and transforms
    idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1_name)]
    idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2_name)]
    t1 = transforms[idx1]
    t2 = transforms[idx2]
    
    # Load tile data
    tile1_data = io_utils.load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = io_utils.load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
    # Apply Z transform to the data
    scale = int(2**pyramid_level)
    z_offset1 = int(round(t1[2, 3] / scale))  # Get Z offset from transform
    z_offset2 = int(round(t2[2, 3] / scale))  # Get Z offset from transform
    
    # Pad or crop the data based on z offsets
    max_z = max(tile1_data.shape[2] + abs(z_offset1), tile2_data.shape[2] + abs(z_offset2))
    min_z = min(0, z_offset1, z_offset2)
    
    # Create padded arrays (initialized to black)
    tile1_padded = np.zeros((tile1_data.shape[0], tile1_data.shape[1], max_z - min_z), dtype=tile1_data.dtype)
    tile2_padded = np.zeros((tile2_data.shape[0], tile2_data.shape[1], max_z - min_z), dtype=tile2_data.dtype)
    
    # Fill padded arrays (rest remains black)
    z1_start = abs(min_z) + z_offset1
    z2_start = abs(min_z) + z_offset2
    tile1_padded[:, :, z1_start:z1_start + tile1_data.shape[2]] = tile1_data
    tile2_padded[:, :, z2_start:z2_start + tile2_data.shape[2]] = tile2_data
    
    if z_slice is None:
        z_slice = (max_z - min_z) // 2
    elif z_slice =="max":
        # determine which z slice had the most signal together
        z_slice = np.argmax(np.sum(tile1_padded, axis=(0,1)) + np.sum(tile2_padded, axis=(0,1)))
    elif z_slice == "center":
        z_slice = (max_z - min_z) // 2
    
    # # Create RGB arrays for visualization
    # def create_rgb_slice(data, z_idx, color):
    #     """Create RGB array with data in specified color channel"""
    #     rgb = np.zeros((*data.shape[:2], 3))
    #     if color == 'red':
    #         rgb[..., 0] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
    #     elif color == 'green':
    #         rgb[..., 1] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
    #     return np.clip(rgb, 0, 1)


        # Create RGB arrays for visualization
    def create_rgb_slice(data, z_idx, color, clip_range=None):
        """Create RGB array with data in specified color channel"""
        rgb = np.zeros((*data.shape[:2], 3))

        # clip to percentile 1 and 99 of whole stack
        if clip_range is not None:
            min_val, max_val = clip_range
        else:
            min_val = np.percentile(data, 1)
            max_val = np.percentile(data, 99.9)
        data = np.clip(data, min_val, max_val)
        print(min_val, max_val)
        if color == 'red':
            slice_data = data[:, :, z_idx]
            rgb[..., 0] = slice_data / max_val
        elif color == 'green':
            slice_data = data[:, :, z_idx]
            rgb[..., 1] = slice_data / max_val
        return np.clip(rgb, 0, 1)
    
    # Create RGB arrays using padded data
    rgb1 = create_rgb_slice(tile1_padded, z_slice, 'red')
    rgb2 = create_rgb_slice(tile2_padded, z_slice, 'green')
    
    # Calculate transformed coordinates
    def get_transformed_coords(data_shape, transform):
        """Get transformed corner coordinates relative to center (0,0)"""
        h, w = data_shape[:2]
        # Define corners relative to center of tile
        corners = np.array([
            [-w/2, -h/2, 0, 1],  # top-left
            [w/2, -h/2, 0, 1],   # top-right
            [-w/2, h/2, 0, 1],   # bottom-left
            [w/2, h/2, 0, 1]     # bottom-right
        ])
        transformed = np.dot(transform, corners.T).T
        return transformed[:, [0, 1]]  # Return x,y coordinates
    
    # Scale transforms by pyramid level
    scale = 2**pyramid_level
    t1_scaled = t1.copy()
    t2_scaled = t2.copy()
    t1_scaled[:3, 3] /= scale
    t2_scaled[:3, 3] /= scale
    
    coords1 = get_transformed_coords(tile1_data.shape, t1_scaled)
    coords2 = get_transformed_coords(tile2_data.shape, t2_scaled)
    
    # Create a combined image that covers both tiles
    x_coords = np.concatenate([coords1[:, 0], coords2[:, 0]])
    y_coords = np.concatenate([coords1[:, 1], coords2[:, 1]])
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Calculate pixel dimensions needed for the combined image
    pixel_size = 1.0  # base pixel size
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)
    
    # Create empty combined image
    combined = np.zeros((height, width, 3))
    
    # Map tile coordinates to combined image coordinates
    def map_to_combined(rgb, coords):
        src_h, src_w = rgb.shape[:2]
        x_start = int((coords[:, 0].min() - x_min) / pixel_size)
        y_start = int((coords[:, 1].min() - y_min) / pixel_size)
        x_end = x_start + src_w
        y_end = y_start + src_h
        return (slice(y_start, y_end), slice(x_start, x_end))
    
    # Add both tiles to the combined image
    slice1 = map_to_combined(rgb1, coords1)
    slice2 = map_to_combined(rgb2, coords2)
    combined[slice1] += rgb1
    combined[slice2] += rgb2
    
    # Clip to ensure we don't exceed 1.0
    combined = np.clip(combined, 0, 1)
    
    # Return the combined image and extent for plotting
    extent = [x_min, x_max, y_min, y_max]
    
    return combined, extent, z_slice

def plot_adjacent_tile_pair(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax


def plot_adjacent_tile_pair(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax



def plot_adjacent_tile_pair_zoom(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file, or str 'center' or 'max'
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    # Get the transformed and combined tile data
    combined, extent, slice_ind = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )

    # rotate the combined image 90 degrees if longer in y than x
    if combined.shape[0] > combined.shape[1]:
        combined = np.rot90(combined)
        extent = [extent[2], extent[3], extent[0], extent[1]]
    
    # Create figure with GridSpec for main view, zoom, and top regions
    from matplotlib import gridspec
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 1, 1], wspace=0.05, hspace=0.05)
    
    # Main view
    ax_main = plt.subplot(gs[0])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    ax_main.set_aspect('equal')
    
    # Add tile information
    pos1, ch = parse_tile_name(tile1_name)
    pos2, ch = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Ch: {ch}\nZ-slice: {slice_index} - {slice_ind}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        print(zoom_extent)
        
        # Create zoom view
        ax_zoom = plt.subplot(gs[1])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        # y off
        ax_zoom.set_yticks([])
        
        ax_zoom.set_title('Overlap')
        ax_zoom.set_aspect('auto')
        
        # get crop of overlap region for top and bottom
        overlap_region = brightened[y_min:y_max, x_min:x_max]
        overlap_mask_cropped = overlap_mask[y_min:y_max, x_min:x_max]

        # get the top and bottom halves of the overlap region
        h_mid = (y_max - y_min) // 2
        top_region = overlap_region[:h_mid, :]
        bottom_region = overlap_region[h_mid:, :]


        overlap_start = np.where(overlap_mask_cropped)[1].min()
        overlap_end = np.where(overlap_mask_cropped)[1].max()
        overlap_start = max(0, overlap_start - 0.8 * (overlap_end - overlap_start))
        overlap_end = min(bottom_region.shape[1], overlap_end + 0.8 * (overlap_end - overlap_start))

        # create a subplot for the top half
        ax_top = plt.subplot(gs[2])
        ax_top.set_facecolor('black')
        ax_top.imshow(top_region)
        ax_top.set_xlim(overlap_start, overlap_end)
        ax_top.set_aspect('auto')
        ax_top.set_title('Overlap Top')
        # y off
        ax_top.set_yticks([])

        # create a subplot for the bottom half
        ax_bottom = plt.subplot(gs[3])
        ax_bottom.set_facecolor('black')
        ax_bottom.imshow(bottom_region)
        ax_bottom.set_xlim(overlap_start, overlap_end)
        ax_bottom.set_aspect('auto')
        ax_bottom.set_title('Overlap Bottom')
        # y off
        ax_bottom.set_yticks([])

    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{pos1}_{pos2}_ch{ch}_z{slice_index}_pyr{pyramid_level}.png'
        
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_zoom2(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file, or str 'center' or 'max'
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    sns.set_context("talk")
    # Get the transformed and combined tile data
    combined, extent, slice_ind = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )

    # rotate the combined image 90 degrees if longer in y than x
    if combined.shape[0] > combined.shape[1]:
        combined = np.rot90(combined)
        extent = [extent[2], extent[3], extent[0], extent[1]]
    
    # Create figure with GridSpec for main view on top, zoom and sections below
    from matplotlib import gridspec
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2.5], width_ratios=[1, 1, 1])
    
    # Main view (spans all columns in top row)
    ax_main = plt.subplot(gs[0, :])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    ax_main.set_aspect('equal')
    
    # Add tile information
    pos1, ch = parse_tile_name(tile1_name)
    pos2, ch = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Ch: {ch}\nZ-slice: {slice_index} - {slice_ind}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        print(zoom_extent)
        
        # Create zoom view in bottom left
        ax_zoom = plt.subplot(gs[1, 0])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        # y off
        ax_zoom.set_yticks([])
        
        ax_zoom.set_title('Overlap')
        ax_zoom.set_aspect('auto')
        
        # Draw a rectangle on the main plot showing the zoom region
        rect = plt.Rectangle(
            (zoom_extent[0], zoom_extent[2]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[2],
            linewidth=1, edgecolor='white', facecolor='none'
        )
        ax_main.add_patch(rect)
        
        # get crop of overlap region for top and bottom
        overlap_region = brightened[y_min:y_max, x_min:x_max]
        overlap_mask_cropped = overlap_mask[y_min:y_max, x_min:x_max]

        # get the top and bottom halves of the overlap region
        h_mid = (y_max - y_min) // 2
        top_region = overlap_region[:h_mid, :]
        bottom_region = overlap_region[h_mid:, :]

        overlap_start = np.where(overlap_mask_cropped)[1].min()
        overlap_end = np.where(overlap_mask_cropped)[1].max()
        overlap_start = max(0, overlap_start - 0.8 * (overlap_end - overlap_start))
        overlap_end = min(bottom_region.shape[1], overlap_end + 0.8 * (overlap_end - overlap_start))

        # create a subplot for the top half in bottom middle
        ax_top = plt.subplot(gs[1, 1])
        ax_top.set_facecolor('black')
        ax_top.imshow(top_region)
        ax_top.set_xlim(overlap_start, overlap_end)
        ax_top.set_aspect('auto')
        ax_top.set_title('Overlap Top')
        # y off
        ax_top.set_yticks([])

        # create a subplot for the bottom half in bottom right
        ax_bottom = plt.subplot(gs[1, 2])
        ax_bottom.set_facecolor('black')
        ax_bottom.imshow(bottom_region)
        ax_bottom.set_xlim(overlap_start, overlap_end)
        ax_bottom.set_aspect('auto')
        ax_bottom.set_title('Overlap Bottom')
        # y off
        ax_bottom.set_yticks([])

    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{pos1}_{pos2}_ch{ch}_z{slice_index}_pyr{pyramid_level}.png'
        
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_zoom3(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure with GridSpec for main view, zoom, and vertical sections
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])
    
    # Main view
    ax_main = plt.subplot(gs[0])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        
        # Create zoom view
        ax_zoom = plt.subplot(gs[1])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        ax_zoom.set_title('Overlap Region (Zoomed)')
        
        # Add a small grid to the zoom view
        ax_zoom.grid(True, color='white', alpha=0.2, linestyle=':')
        
        # Create a nested GridSpec for the 2x2 vertical sections
        gs_sections = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2])
        
        # Calculate the height of each vertical section
        section_height = (y_max - y_min) // 4
        
        # Create 4 vertical sections
        for i in range(4):
            row = i // 2
            col = i % 2
            
            # Calculate the vertical bounds for this section
            section_y_min = y_min + i * section_height
            section_y_max = section_y_min + section_height
            
            # Ensure the last section goes to the end
            if i == 3:
                section_y_max = y_max
                
            # Calculate the extent of this section
            section_extent = [
                zoom_extent[0],
                zoom_extent[1],
                extent[2] + section_y_min * pixel_height,
                extent[2] + section_y_max * pixel_height
            ]
            
            # Create subplot
            ax_section = plt.subplot(gs_sections[row, col])
            ax_section.set_facecolor('black')
            
            # Plot the section
            ax_section.imshow(brightened, extent=extent)
            ax_section.set_xlim(section_extent[0], section_extent[1])
            ax_section.set_ylim(section_extent[2], section_extent[3])
            
            # Turn off axis
            ax_section.set_xticks([])
            ax_section.set_yticks([])
            
            # Draw a rectangle on the zoom view showing this section
            rect = plt.Rectangle(
                (section_extent[0], section_extent[2]),
                section_extent[1] - section_extent[0],
                section_extent[3] - section_extent[2],
                linewidth=1, edgecolor=['red', 'green', 'blue', 'cyan'][i], facecolor='none'
            )
            ax_zoom.add_patch(rect)
            
            # Add a small colored marker in the corner of each section to match the rectangle
            ax_section.plot(section_extent[0] + 5*pixel_width, section_extent[2] + 5*pixel_height, 
                          'o', color=['red', 'green', 'blue', 'cyan'][i], markersize=8)
        
        # Add a title to the sections subplot area
        plt.figtext(0.83, 0.95, 'Vertical Sections', ha='center', va='center', fontsize=12)
        
    else:
        # If no overlap is found, just display a message
        for ax_idx in [1, 2]:
            ax = plt.subplot(gs[ax_idx])
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, 'No overlap detected', 
                    color='white', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            if ax_idx == 1:
                ax.set_title('Overlap Region')
            else:
                ax.set_title('Vertical Sections')
    
    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_t(tile1_name, tile2_name, 
                              transforms, tile_names, bucket_name, dataset_path, 
                          slice=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice: slice in 
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {z_slice}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{z_slice}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax

def plot_all_adjacent_pairs(pairs, transforms, tile_names, bucket_name, dataset_path,
                           z_slice=None, pyramid_level=0, save=False, output_dir=None,
                           max_pairs=None):
    """
    Plot all adjacent tile pairs
    
    Args:
        pairs: List of (tile1, tile2) tuples from get_all_adjacent_pairs()
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Optional z-slice index. If None, uses middle slice
        pyramid_level: Pyramid level to load (default 0 = full resolution)
        save: If True, saves figures instead of displaying
        output_dir: Directory to save figures if save=True
        max_pairs: Maximum number of pairs to plot (None for all)
    """
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    
    for tile1, tile2 in pairs:
        plot_adjacent_tile_pair(
            tile1, tile2, transforms, tile_names,
            bucket_name, dataset_path,
            z_slice=z_slice,
            pyramid_level=pyramid_level,
            save=save,
            output_dir=output_dir
        )



def plot_tile_comparison(tile1_name, tile2_name, bucket_name, dataset_path, 
                        tile_dict, pyramid_level=0, z_slice=100,
                        minmax_percentile=(1,99)):
    """
    Plot two tiles side by side with their position in the grid.
    
    Args:
        tile1_name: name or identifier for first tile
        tile2_name: name or identifier for second tile
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        tile_dict: Dictionary of all tiles {tile_id: tile_name}
        pyramid_level: pyramid level to load (default 0 = full resolution)
        z_slice: z-slice index, or indices for projection
        minmax_percentile: tuple of (min, max) percentiles for contrast adjustment
    """
    tile1_data = io_utils.load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = io_utils.load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)

    print(tile1_data.shape)
    print(tile2_data.shape)

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot first two subplots (individual tiles) as before

    if isinstance(z_slice, tuple):
        tile1_img = np.max(tile1_data[:,:,z_slice], axis=2)
        tile2_img = np.max(tile2_data[:,:,z_slice], axis=2)
    else:
        tile1_img = tile1_data[:,:,z_slice]
        tile2_img = tile2_data[:,:,z_slice]

    # Plot first tile (red)
    vmin = np.percentile(tile1_img, minmax_percentile[0])
    vmax = np.percentile(tile1_img, minmax_percentile[1])
    ax1.imshow(tile1_img, cmap='Reds', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Tile 1 (Red)\n{tile1_name}\nminmax:({vmin}, {vmax})')
    #ax1.grid(True)
    
    # Plot second tile (green)
    vmin = np.percentile(tile2_img, minmax_percentile[0])
    vmax = np.percentile(tile2_img, minmax_percentile[1])
    ax2.imshow(tile2_img, cmap='Greens', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Tile 2 (Green)\n{tile2_name}\nminmax:({vmin}, {vmax})')
    #ax2.grid(True)

    # Create coverage map for third subplot
    coords = []
    for _, tile in tile_dict.items():
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        x = int(parts[2])
        y = int(parts[4])
        coords.append((x, y))
    
    # Find dimensions
    x_coords = {x for x, _ in coords}
    y_coords = {y for _, y in coords}
    
    x_dim = max(x_coords) + 1
    y_dim = max(y_coords) + 1
    
    # Create coverage map
    coverage = np.zeros((y_dim, x_dim))
    for x, y in coords:
        coverage[y, x] = 1

    # Get coordinates of our two tiles
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    
    # Highlight the two tiles we're comparing
    coverage[pos1[1], pos1[0]] = 2  # Mark first tile
    coverage[pos2[1], pos2[0]] = 3  # Mark second tile

    # Plot the coverage map
    im = ax3.imshow(coverage, cmap='RdYlBu', interpolation='nearest')
    ax3.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.set_title('Tile Grid Position')
    
    # Add coordinate labels
    for i in range(x_dim):
        for j in range(y_dim):
            color = 'white' if coverage[j, i] == 0 else 'black'
            ax3.text(i, j, f'({i},{j})', ha='center', va='center', color=color)

    plt.colorbar(im, ax=ax3, label='Tile Present')
    # add a bit of space for subtitle
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f'Tile Pair Comparison Z-slice: {z_slice}') 
    plt.tight_layout()
    plt.show()

def plot_tiles_on_grid(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path,
                      n_x_tiles, n_y_tiles, z_slice=None, pyramid_level=0,
                      color1='red', color2='green'):
    """
    Plot two tiles transformed onto their position in the full nominal grid.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        n_x_tiles, n_y_tiles: Number of tiles in x and y dimensions
        z_slice: Z-slice to display (default: middle slice)
        pyramid_level: Pyramid level to load (default: 0 = full resolution)
        color1, color2: Colors for the two tiles (default: red and green)
    """
    # Load tile data
    tile1_data = io_utils.load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = io_utils.load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
    if z_slice is None:
        z_slice = tile1_data.shape[2] // 2
        
    # Get tile dimensions
    tile_height, tile_width = tile1_data.shape[:2]
    
    # Create full canvas
    full_height = tile_height * n_y_tiles
    full_width = tile_width * n_x_tiles
    canvas = np.full((full_height, full_width, 3), np.nan)
    
    # Get transforms
    idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1_name)]
    idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2_name)]
    t1 = transforms[idx1]
    t2 = transforms[idx2]
    
    # Scale transforms by pyramid level
    scale = 2**pyramid_level
    t1_scaled = t1.copy()
    t2_scaled = t2.copy()
    t1_scaled[:3, 3] /= scale
    t2_scaled[:3, 3] /= scale
    
    def transform_coordinates(y, x, transform):
        """Transform pixel coordinates"""
        coords = np.array([0, y, x, 1])
        transformed = np.dot(transform, coords)
        return transformed[2], transformed[1]  # return x, y
    
    # Create color maps for each tile
    def create_color_array(data, color):
        """Create RGB array with data in specified color channel"""
        rgb = np.zeros((*data.shape[:2], 3))
        if color == 'red':
            rgb[..., 0] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        elif color == 'green':
            rgb[..., 1] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        elif color == 'blue':
            rgb[..., 2] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        return np.clip(rgb, 0, 1)
    
    # Create meshgrid for pixel coordinates
    y, x = np.mgrid[0:tile_height, 0:tile_width]
    
    # Transform and place first tile
    rgb1 = create_color_array(tile1_data, color1)
    for i in range(tile_height):
        for j in range(tile_width):
            y_trans, x_trans = transform_coordinates(i, j, t1_scaled)
            y_idx, x_idx = int(y_trans), int(x_trans)
            if 0 <= y_idx < full_height and 0 <= x_idx < full_width:
                canvas[y_idx, x_idx] = rgb1[i, j]
    
    # Transform and place second tile with blending
    rgb2 = create_color_array(tile2_data, color2)
    for i in range(tile_height):
        for j in range(tile_width):
            y_trans, x_trans = transform_coordinates(i, j, t2_scaled)
            y_idx, x_idx = int(y_trans), int(x_trans)
            if 0 <= y_idx < full_height and 0 <= x_idx < full_width:
                if np.all(np.isnan(canvas[y_idx, x_idx])):
                    canvas[y_idx, x_idx] = rgb2[i, j]
                else:
                    # Blend colors in overlap regions
                    canvas[y_idx, x_idx] = np.nanmax([canvas[y_idx, x_idx], rgb2[i, j]], axis=0)
    
    # Plot the result
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(canvas)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair on Nominal Grid\n{color1}: {pos1} | {color2}: {pos2}\nZ-slice: {z_slice}')
    
    # Add grid lines at tile boundaries
    for i in range(n_x_tiles + 1):
        ax.axvline(x=i * tile_width, color='white', alpha=0.3)
    for i in range(n_y_tiles + 1):
        ax.axhline(y=i * tile_height, color='white', alpha=0.3)
    
    plt.show()
    return fig, ax



def plot_tile_orthogonal_views(tile_data, tile_name, 
                             center_points=None, save=False, output_dir=None):
    """
    Plot orthogonal views (XY, XZ, YZ) through a tile at specified center points.
    
    Args:
        tile_data: 3D numpy array containing the tile data
        tile_name: Name of tile (for display purposes)
        center_points: Dict with 'x', 'y', 'z' keys specifying slice centers. If None, uses middle of volume
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get tile dimensions
    h, w, d = tile_data.shape
    
    # If no center points provided, use middle of volume
    if center_points is None:
        center_points = {
            'x': w // 2,
            'y': h // 2,
            'z': d // 2
        }
    
    # Create figure with three subplots
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY view (top down)
    xy_slice = tile_data[:, :, center_points['z']]
    ax_xy.imshow(xy_slice)
    ax_xy.set_title(f'XY Slice (Z={center_points["z"]})')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    
    # XZ view (side view)
    xz_slice = tile_data[:, center_points['y'], :].T
    ax_xz.imshow(xz_slice)
    ax_xz.set_title(f'XZ Slice (Y={center_points["y"]})')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    
    # YZ view (front view)
    yz_slice = tile_data[center_points['x'], :, :].T
    ax_yz.imshow(yz_slice)
    ax_yz.set_title(f'YZ Slice (X={center_points["x"]})')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    
    # Add tile information to overall figure
    pos = parse_tile_name(tile_name)
    fig.suptitle(f'Orthogonal Views of Tile {pos}')
    
    # Adjust layout
    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'orthogonal_views_{tile_name}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, (ax_xy, ax_xz, ax_yz)

def calculate_accutance(image_slice, percentile_threshold=99):
    """
    Calculate accutance (edge sharpness) for an image slice.
    Uses Sobel operator to detect edges and measures their strength.
    
    Args:
        image_slice: 2D numpy array containing the image slice
        percentile_threshold: Percentile threshold for edge detection (default 99)
        
    Returns:
        dict containing:
            mean_accutance: Mean edge strength
            max_accutance: Maximum edge strength
            accutance_map: 2D array of edge strengths
            edge_mask: Boolean mask of detected edges
    """
    from scipy import ndimage
    
    # Normalize image to [0,1]
    img_norm = image_slice.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Calculate gradients using Sobel operator
    grad_x = ndimage.sobel(img_norm, axis=1)
    grad_y = ndimage.sobel(img_norm, axis=0)
    
    # Calculate gradient magnitude
    accutance_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge mask using threshold
    edge_threshold = np.percentile(accutance_map, percentile_threshold)
    edge_mask = accutance_map > edge_threshold
    
    # Calculate statistics for detected edges
    edge_values = accutance_map[edge_mask]
    mean_accutance = edge_values.mean() if edge_values.size > 0 else 0
    max_accutance = edge_values.max() if edge_values.size > 0 else 0
    
    return {
        'mean_accutance': mean_accutance,
        'max_accutance': max_accutance,
        'accutance_map': accutance_map,
        'edge_mask': edge_mask
    }

def plot_accutance_profile(tile_data, axis='z', slice_range=None, step=1):
    """
    Plot accutance profile along specified axis.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        fig: Figure object
        ax: Axes object
        profile: Dict containing accutance values and positions
    """
    # Set up axis mapping
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    axis_idx = axis_map[axis]
    
    # Determine slice range
    if slice_range is None:
        slice_range = (0, tile_data.shape[axis_idx])
    
    # Initialize arrays for profile
    positions = range(slice_range[0], slice_range[1], step)
    mean_accutance = []
    max_accutance = []
    
    # Calculate accutance for each slice
    for pos in positions:
        # Take slice along specified axis
        if axis == 'z':
            slice_data = tile_data[pos, :, :]
        elif axis == 'y':
            slice_data = tile_data[:, pos, :]
        else:  # x
            slice_data = tile_data[:, :, pos]
            
        # Calculate accutance
        acc = calculate_accutance(slice_data)
        mean_accutance.append(acc['mean_accutance'])
        max_accutance.append(acc['max_accutance'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, mean_accutance, label='Mean Accutance', color='blue')
    ax.plot(positions, max_accutance, label='Max Accutance', color='red', alpha=0.5)
    
    ax.set_xlabel(f'{axis.upper()} Position')
    ax.set_ylabel('Accutance')
    ax.set_title(f'Accutance Profile Along {axis.upper()} Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    profile = {
        'positions': positions,
        'mean_accutance': mean_accutance,
        'max_accutance': max_accutance
    }
    
    return fig, ax, profile

def plot_accutance_comparison(tile1_data, tile2_data, axis='z', slice_range=None, step=1):
    """
    Plot accutance profiles for two tiles side by side.
    
    Args:
        tile1_data, tile2_data: 3D numpy arrays (z,y,x)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        fig: Figure object
        axes: List of axes objects
        profiles: Dict containing accutance profiles for both tiles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate and plot profiles
    _, _, profile1 = plot_accutance_profile(tile1_data, axis, slice_range, step)
    _, _, profile2 = plot_accutance_profile(tile2_data, axis, slice_range, step)
    
    # Plot on first axis
    ax1.plot(profile1['positions'], profile1['mean_accutance'], 
             label='Tile 1 Mean', color='blue')
    ax1.plot(profile1['positions'], profile1['max_accutance'], 
             label='Tile 1 Max', color='blue', alpha=0.5)
    ax1.plot(profile2['positions'], profile2['mean_accutance'], 
             label='Tile 2 Mean', color='red')
    ax1.plot(profile2['positions'], profile2['max_accutance'], 
             label='Tile 2 Max', color='red', alpha=0.5)
    
    ax1.set_xlabel(f'{axis.upper()} Position')
    ax1.set_ylabel('Accutance')
    ax1.set_title('Accutance Profiles Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot difference on second axis
    mean_diff = np.array(profile1['mean_accutance']) - np.array(profile2['mean_accutance'])
    max_diff = np.array(profile1['max_accutance']) - np.array(profile2['max_accutance'])
    
    ax2.plot(profile1['positions'], mean_diff, label='Mean Difference', color='green')
    ax2.plot(profile1['positions'], max_diff, label='Max Difference', color='green', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel(f'{axis.upper()} Position')
    ax2.set_ylabel('Accutance Difference')
    ax2.set_title('Accutance Difference (Tile 1 - Tile 2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    profiles = {
        'tile1': profile1,
        'tile2': profile2,
        'difference': {
            'positions': profile1['positions'],
            'mean_difference': mean_diff,
            'max_difference': max_diff
        }
    }
    
    return fig, [ax1, ax2], profiles

def calculate_block_accutance_profiles(tile_data, n_blocks=3, axis='z', slice_range=None, step=1):
    """
    Calculate accutance profiles for each block in an nxn grid of the tile.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        profiles: Dict containing accutance profiles for each block
        block_bounds: Dict containing block boundaries
    """
    # Get dimensions
    z_dim, y_dim, x_dim = tile_data.shape
    
    # Calculate block sizes
    y_block_size = y_dim // n_blocks
    x_block_size = x_dim // n_blocks
    
    # Calculate block boundaries
    y_bounds = [(i * y_block_size, (i + 1) * y_block_size) for i in range(n_blocks)]
    x_bounds = [(i * x_block_size, (i + 1) * x_block_size) for i in range(n_blocks)]
    
    # Set up axis mapping
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    axis_idx = axis_map[axis]
    
    # Determine slice range
    if slice_range is None:
        slice_range = (0, tile_data.shape[axis_idx])
    
    # Initialize arrays for profiles
    positions = range(slice_range[0], slice_range[1], step)
    profiles = {}
    
    # Calculate accutance for each block
    for i in range(n_blocks):
        for j in range(n_blocks):
            block_id = f'block_{i}_{j}'
            y_start, y_end = y_bounds[i]
            x_start, x_end = x_bounds[j]
            
            mean_accutance = []
            max_accutance = []
            
            # Calculate accutance for each slice
            for pos in positions:
                if axis == 'z':
                    slice_data = tile_data[pos, y_start:y_end, x_start:x_end]
                elif axis == 'y':
                    slice_data = tile_data[:, pos, x_start:x_end]
                else:  # x
                    slice_data = tile_data[:, x_start:x_end, pos]
                
                acc = calculate_accutance(slice_data)
                mean_accutance.append(acc['mean_accutance'])
                max_accutance.append(acc['max_accutance'])
            
            profiles[block_id] = {
                'positions': positions,
                'mean_accutance': mean_accutance,
                'max_accutance': max_accutance,
                'bounds': {
                    'y': (y_start, y_end),
                    'x': (x_start, x_end)
                }
            }
    
    block_bounds = {
        'y': y_bounds,
        'x': x_bounds
    }
    
    return profiles, block_bounds

def plot_block_accutance_profiles(tile_data, tile_name=None, n_blocks=3, axis='z', 
                                slice_range=None, step=1, plot_max=False):
    """
    Plot accutance profiles for each block in an nxn grid of the tile.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        tile_name: Name of tile for title (optional)
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        plot_max: Whether to plot max accutance (default False, plots mean)
        
    Returns:
        fig: Figure object
        ax: Axes object
        profiles: Dict containing accutance profiles for each block
    """
    # Calculate profiles for each block
    profiles, block_bounds = calculate_block_accutance_profiles(
        tile_data, n_blocks, axis, slice_range, step
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot profiles for each block with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, n_blocks * n_blocks))
    
    for idx, (block_id, profile) in enumerate(profiles.items()):
        i, j = map(int, block_id.split('_')[1:])
        label = f'Block ({i},{j})'
        
        if plot_max:
            ax.plot(profile['positions'], profile['max_accutance'], 
                   label=label, color=colors[idx], alpha=0.7)
        else:
            ax.plot(profile['positions'], profile['mean_accutance'], 
                   label=label, color=colors[idx], alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(f'{axis.upper()} Position')
    ax.set_ylabel('Accutance')
    title = f'Block Accutance Profiles Along {axis.upper()} Axis'
    if tile_name:
        pos = parse_tile_name(tile_name)
        title += f'\nTile {pos}'
    if plot_max:
        title += ' (Maximum Values)'
    else:
        title += ' (Mean Values)'
    ax.set_title(title)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax, profiles

def plot_block_accutance_heatmap(tile_data, z_slice, n_blocks=3, percentile_threshold=99, 
                                minmax_lims=(1, 99),
                               normalize_method: str = None):
    """
    Plot a heatmap of accutance values for each block in an nxn grid at a specific z-slice,
    alongside the original image slice with grid overlay and detected edges.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        z_slice: Z-slice index to analyze
        minmax_lims: tuple of (vmin, vmax) to use for image slice
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        percentile_threshold: Threshold for edge detection
        normalize_method: Method to normalize accutance values (default None)
    Returns:
        fig: Figure object
        axes: List of axes objects [ax_image, ax_heatmap, ax_edges]
        accutance_values: 2D array of accutance values
    """
    # Get dimensions
    _, y_dim, x_dim = tile_data.shape
    
    # Calculate block sizes
    y_block_size = y_dim // n_blocks
    x_block_size = x_dim // n_blocks
    
    # Initialize array for accutance values
    accutance_values = np.zeros((n_blocks, n_blocks))
    
    # Create an array to store the full edge mask for visualization
    full_edge_mask = np.zeros((y_dim, x_dim), dtype=bool)
    full_accutance_map = np.zeros((y_dim, x_dim))
    
    # Calculate accutance for each block
    for i in range(n_blocks):
        for j in range(n_blocks):
            y_start = i * y_block_size
            y_end = (i + 1) * y_block_size
            x_start = j * x_block_size
            x_end = (j + 1) * x_block_size
            
            block_data = tile_data[z_slice, y_start:y_end, x_start:x_end]
            
            acc = calculate_normalized_accutance(block_data, percentile_threshold, normalize_method)
            if normalize_method:
                accutance_values[i, j] = acc['normalized_mean_accutance']
            else:
                accutance_values[i, j] = acc['raw_mean_accutance']
            
            # Store edge mask and accutance map for visualization
            full_edge_mask[y_start:y_end, x_start:x_end] = acc['edge_mask']
            full_accutance_map[y_start:y_end, x_start:x_end] = acc['accutance_map']
    
    # Create figure with three subplots
    sns.set_context('talk')
    fig, (ax_image, ax_heatmap, ax_edges) = plt.subplots(1, 3, figsize=(24, 7))
    
    # Plot original image
    image_slice = tile_data[z_slice]
    if minmax_lims is None:
        im_image = ax_image.imshow(image_slice, cmap='gray')
    else:
        vmin, vmax = np.percentile(image_slice, minmax_lims)
        im_image = ax_image.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax_image.set_title(f'Original Image\nZ-slice {z_slice}')
    
    # Add grid lines to original image
    for i in range(1, n_blocks):
        ax_image.axhline(y=i * y_block_size, color='r', linestyle='--', alpha=0.5)
        ax_image.axvline(x=i * x_block_size, color='r', linestyle='--', alpha=0.5)
    
    # # Add block numbers to original image
    # for i in range(n_blocks):
    #     for j in range(n_blocks):
    #         center_y = (i + 0.5) * y_block_size
    #         center_x = (j + 0.5) * x_block_size
    #         ax_image.text(center_x, center_y, f'({i},{j})', 
    #                     ha='center', va='center', 
    #                     color='red', fontweight='bold',
    #                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot heatmap
    im_heatmap = ax_heatmap.imshow(accutance_values, cmap='magma')
    ax_heatmap.set_title(f'Accutance Heatmap\nNormalized: {normalize_method}')
    
    # Add colorbar to heatmap
    plt.colorbar(im_heatmap, ax=ax_heatmap, label='Accutance')
    
    # Add block labels to heatmap
    for i in range(n_blocks):
        for j in range(n_blocks):
            text = f'{accutance_values[i, j]:.2f}'
            # add text with a white background
            ax_heatmap.text(j, i, text, ha='center', va='center', 
                          color='black', fontweight='bold', fontsize=18,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add labels to heatmap
    ax_heatmap.set_xticks(range(n_blocks))
    ax_heatmap.set_yticks(range(n_blocks))
    ax_heatmap.set_xlabel('X Block')
    ax_heatmap.set_ylabel('Y Block')
    
    # Add grid lines to heatmap
    ax_heatmap.grid(True, color='white', linestyle='-', alpha=0.2)
    
    # Plot edge visualization
    # Create an overlay with original image and detected edges
    edge_overlay = np.zeros((y_dim, x_dim, 3))
    # Add grayscale image to all channels
    for c in range(3):
        edge_overlay[:, :, c] = image_slice / np.max(image_slice) if np.max(image_slice) > 0 else 0
    
    # Highlight edges in red
    edge_overlay[full_edge_mask, 0] = 1.0  # Red channel
    edge_overlay[full_edge_mask, 1] = 0.0  # Green channel
    edge_overlay[full_edge_mask, 2] = 0.0  # Blue channel
    
    ax_edges.imshow(edge_overlay)
    ax_edges.set_title(f'Detected Edges\nPercentile: {percentile_threshold}')
    
    # Add grid lines to edge visualization
    for i in range(1, n_blocks):
        ax_edges.axhline(y=i * y_block_size, color='cyan', linestyle='--', alpha=0.5)
        ax_edges.axvline(x=i * x_block_size, color='cyan', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, [ax_image, ax_heatmap, ax_edges], accutance_values

####
# All tiles accutance
####
def calculate_tile_grid_block_accutance(tile_dict, bucket_name, dataset_path,
                                      z_slice, n_blocks=3, pyramid_level=0, percentile_threshold=99):
    """
    Calculate accutance values for 3x3 blocks within each tile across the entire tile grid.
    
    Args:
        tile_dict: Dictionary mapping tile IDs to tile names
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Z-slice to analyze
        n_blocks: Number of blocks per tile dimension (default 3)
        pyramid_level: Pyramid level to load
        percentile_threshold: Threshold for edge detection
        
    Returns:
        dict containing:
            grid_accutance: 2D array of shape (grid_y * n_blocks, grid_x * n_blocks) with accutance values
            tile_positions: Dictionary mapping tile IDs to their grid positions
            coverage_map: 2D boolean array showing tile presence
    """
    # First analyze the tile grid to get dimensions and positions
    grid_info = analyze_tile_grid(tile_dict, plot=False)
    grid_x, grid_y = grid_info['dimensions'][:2]
    coverage_map = grid_info['coverage_map']
    
    # Initialize the full accutance grid
    full_grid = np.full((grid_y * n_blocks, grid_x * n_blocks), np.nan)
    
    # Process each tile
    for tile_id, tile_name in tile_dict.items():
        print(f'Processing tile {tile_name}')
        # Extract tile position from name
        parts = tile_name.split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        # Load and process tile
        tile_data = io_utils.load_tile_data(tile_name, bucket_name, dataset_path, pyramid_level)
        
        # Calculate accutance for each block in the tile
        y_block_size = tile_data.shape[0] // n_blocks
        x_block_size = tile_data.shape[1] // n_blocks
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                y_start = i * y_block_size
                y_end = (i + 1) * y_block_size
                x_start = j * x_block_size
                x_end = (j + 1) * x_block_size
                
                block_data = tile_data[x_start:x_end, y_start:y_end, z_slice]
                acc = calculate_accutance(block_data, percentile_threshold)
                
                # Calculate position in full grid
                grid_y_pos = tile_y * n_blocks + i
                grid_x_pos = tile_x * n_blocks + j
                
                full_grid[grid_y_pos, grid_x_pos] = acc['mean_accutance']
    
    return {
        'grid_accutance': full_grid,
        'coverage_map': coverage_map,
        'dimensions': (grid_x, grid_y),
        'blocks_per_tile': n_blocks
    }

def plot_tile_grid_block_accutance(grid_data, z_slice, show_tile_boundaries=True):
    """
    Plot the accutance heatmap for all tiles with their block subdivisions.
    
    Args:
        grid_data: Output from calculate_tile_grid_block_accutance
        z_slice: Z-slice being displayed
        show_tile_boundaries: Whether to show tile boundaries
        
    Returns:
        fig: Figure object
        ax: Axes object
    """
    grid_accutance = grid_data['grid_accutance']
    coverage_map = grid_data['coverage_map']
    grid_x, grid_y = grid_data['dimensions']
    n_blocks = grid_data['blocks_per_tile']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot heatmap
    im = ax.imshow(grid_accutance, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Mean Accutance')
    
    # Add tile boundaries
    if show_tile_boundaries:
        for i in range(grid_y):
            ax.axhline(y=(i+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
        for j in range(grid_x):
            ax.axvline(x=(j+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add block grid lines
    for i in range(grid_accutance.shape[0]):
        ax.axhline(y=i-0.5, color='white', linestyle='-', alpha=0.1)
    for j in range(grid_accutance.shape[1]):
        ax.axvline(x=j-0.5, color='white', linestyle='-', alpha=0.1)
    
    # Add labels
    ax.set_title(f'Tile Grid Block Accutance Map (Z-slice {z_slice})')
    ax.set_xlabel('X Position (Blocks)')
    ax.set_ylabel('Y Position (Blocks)')
    
    # Add text showing accutance values
    for i in range(grid_accutance.shape[0]):
        for j in range(grid_accutance.shape[1]):
            if not np.isnan(grid_accutance[i, j]):
                text = f'{grid_accutance[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax

def plot_tile_grid_block_accutance_with_image(tile_dict, transforms, tile_names, 
                                             bucket_name, dataset_path, z_slice, 
                                             n_blocks=3, pyramid_level=0):
    """
    Plot both the original stitched image and the accutance heatmap side by side.
    
    Args:
        ... (same as calculate_tile_grid_block_accutance) ...
        
    Returns:
        fig: Figure object
        axes: List of axes objects [ax_image, ax_heatmap]
    """
    # Calculate accutance values
    grid_data = calculate_tile_grid_block_accutance(
        tile_dict, transforms, tile_names, bucket_name, dataset_path,
        z_slice, n_blocks, pyramid_level
    )
    
    # Create figure with two subplots
    fig, (ax_image, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot stitched image
    # (You'll need to implement this part based on your stitching functionality)
    # For now, we'll just show the coverage map
    ax_image.imshow(grid_data['coverage_map'], cmap='gray')
    ax_image.set_title(f'Tile Coverage Map\nZ-slice {z_slice}')
    
    # Plot heatmap
    im = ax_heatmap.imshow(grid_data['grid_accutance'], cmap='viridis')
    plt.colorbar(im, ax=ax_heatmap, label='Mean Accutance')
    
    # Add tile boundaries
    grid_x, grid_y = grid_data['dimensions']
    n_blocks = grid_data['blocks_per_tile']
    
    for i in range(grid_y):
        ax_heatmap.axhline(y=(i+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    for j in range(grid_x):
        ax_heatmap.axvline(x=(j+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add labels
    ax_heatmap.set_title('Block Accutance Heatmap')
    ax_heatmap.set_xlabel('X Position (Blocks)')
    ax_heatmap.set_ylabel('Y Position (Blocks)')
    
    plt.tight_layout()
    return fig, [ax_image, ax_heatmap]

def calculate_normalized_accutance(image_slice, percentile_threshold=99, normalization_method='edge_density'):
    """
    Calculate accutance (edge sharpness) normalized for feature density.
    
    Args:
        image_slice: 2D numpy array containing the image slice
        percentile_threshold: Percentile threshold for edge detection (default 99)
        normalization_method: Method for normalization
            - 'edge_density': Normalize by the percentage of pixels that are edges
            - 'edge_count': Normalize by the absolute count of edge pixels
            - 'local_contrast': Use local contrast normalization before edge detection
            - 'structure_tensor': Use eigenvalues of structure tensor approach
        
    Returns:
        dict containing normalized and raw accutance metrics
    """
    from scipy import ndimage
    
    # Normalize image to [0,1]
    img_norm = image_slice.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Calculate gradients using Sobel operator
    grad_x = ndimage.sobel(img_norm, axis=1)
    grad_y = ndimage.sobel(img_norm, axis=0)
    
    # Calculate gradient magnitude
    accutance_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge mask using threshold
    edge_threshold = np.percentile(accutance_map, percentile_threshold)
    edge_mask = accutance_map > edge_threshold
    edge_count = np.sum(edge_mask)
    total_pixels = edge_mask.size
    edge_density = edge_count / total_pixels if total_pixels > 0 else 0
    
    # Calculate statistics for detected edges
    edge_values = accutance_map[edge_mask]
    raw_mean_accutance = edge_values.mean() if edge_values.size > 0 else 0
    raw_max_accutance = edge_values.max() if edge_values.size > 0 else 0
    
    # Normalize based on selected method
    if normalization_method == 'edge_density':
        # Normalize by edge density - adjusts for different feature densities
        # Areas with few edges but sharp will score higher than areas with many edges but blurry
        norm_factor = max(0.001, edge_density)  # Avoid division by zero
        normalized_mean = raw_mean_accutance / norm_factor
        normalized_max = raw_max_accutance / norm_factor
        
    elif normalization_method == 'edge_count':
        # Use logarithmic scaling based on edge count
        norm_factor = max(1, np.log10(edge_count + 1))
        normalized_mean = raw_mean_accutance / norm_factor
        normalized_max = raw_max_accutance / norm_factor
        
    elif normalization_method == 'local_contrast':
        # This method needs to be applied before edge detection
        # Use local contrast normalization before calculating accutance
        local_mean = ndimage.uniform_filter(img_norm, size=15)
        local_std = np.sqrt(ndimage.uniform_filter(img_norm**2, size=15) - local_mean**2)
        local_std = np.maximum(local_std, 0.0001)  # Avoid division by zero
        normalized_img = (img_norm - local_mean) / local_std
        
        # Recalculate edges on contrast-normalized image
        norm_grad_x = ndimage.sobel(normalized_img, axis=1)
        norm_grad_y = ndimage.sobel(normalized_img, axis=0)
        norm_accutance_map = np.sqrt(norm_grad_x**2 + norm_grad_y**2)
        
        # Use same edge threshold approach
        norm_edge_threshold = np.percentile(norm_accutance_map, percentile_threshold)
        norm_edge_mask = norm_accutance_map > norm_edge_threshold
        norm_edge_values = norm_accutance_map[norm_edge_mask]
        
        normalized_mean = norm_edge_values.mean() if norm_edge_values.size > 0 else 0
        normalized_max = norm_edge_values.max() if norm_edge_values.size > 0 else 0
        
    elif normalization_method == 'structure_tensor':
        # Structure tensor approach (based on Harris corner detector)
        # This weights edge strength by local structure importance
        gaussian_filter = lambda x, sigma: ndimage.gaussian_filter(x, sigma)
        gx2 = gaussian_filter(grad_x * grad_x, 1.5)
        gy2 = gaussian_filter(grad_y * grad_y, 1.5)
        gxy = gaussian_filter(grad_x * grad_y, 1.5)
        
        # Eigenvalues represent strength of edge/corner response
        # For each pixel, calculate the trace and determinant
        trace = gx2 + gy2
        det = gx2 * gy2 - gxy * gxy
        
        # Calculate eigenvalues (λ1 ≥ λ2)
        # Using: λ1,λ2 = (trace/2) ± sqrt((trace/2)^2 - det)
        trace_half = trace / 2
        discriminant = np.sqrt(np.maximum(0, trace_half**2 - det))
        lambda1 = trace_half + discriminant  # Larger eigenvalue
        
        # Use largest eigenvalue as corner/edge strength measure
        structure_strength = lambda1
        normalized_mean = np.mean(structure_strength)
        normalized_max = np.max(structure_strength)
    
    else:
        normalized_mean = raw_mean_accutance
        normalized_max = raw_max_accutance
    
    return {
        'raw_mean_accutance': raw_mean_accutance,
        'raw_max_accutance': raw_max_accutance,
        'normalized_mean_accutance': normalized_mean,
        'normalized_max_accutance': normalized_max,
        'edge_density': edge_density,
        'edge_count': edge_count,
        'accutance_map': accutance_map,
        'edge_mask': edge_mask,
        'normalization_method': normalization_method
    }
