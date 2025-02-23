import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from pathlib import Path

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
    return x, y, z

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
    x, y, z = parse_tile_name(tile_name)
    
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

def plot_adjacent_tile_pair(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          z_slice=None, pyramid_level=0, save=False, output_dir=None):
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
        z_slice: Z-slice to display
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get indices and transforms
    idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1_name)]
    idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2_name)]
    t1 = transforms[idx1]
    t2 = transforms[idx2]
    
    # Load tile data
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
    # Apply Z transform to the data
    scale = int(2**pyramid_level)
    z_offset1 = int(round(t1[2, 3] / scale))  # Get Z offset from transform
    z_offset2 = int(round(t2[2, 3] / scale))  # Get Z offset from transform
    
    print(f"Z offset 1: {z_offset1}")
    print(f"Z offset 2: {z_offset2}")
    
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
    
    # Create RGB arrays for visualization
    def create_rgb_slice(data, z_idx, color):
        """Create RGB array with data in specified color channel"""
        rgb = np.zeros((*data.shape[:2], 3))
        if color == 'red':
            rgb[..., 0] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
        elif color == 'green':
            rgb[..., 1] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    
    # Plot combined image
    ax.imshow(combined, extent=[x_min, x_max, y_min, y_max])
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {z_slice}')
    
    # # Add grid and center marker
    # ax.grid(True, alpha=0.3)
    # ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    # ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    # ax.plot(0, 0, 'w+', markersize=10, label='Grid Center (0,0)')
    
    # Set axis limits with padding
    padding = 0.05  # 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)
    ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
    
    # Add legend
    #ax.legend()
    
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


def load_tile_data(tile_name, bucket_name, dataset_path, pyramid_level=0,dims_order=(1,2,0)):
    """
    Load tile data from zarr
    """
    tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
    zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
    tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
    return tile_data.compute().transpose(dims_order)

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
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)

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
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
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