import torch
from torch.nn import functional as F
import numpy as np

from stitch_utils import analyze_tile_grid, load_tile_data

def calculate_tile_grid_block_accutance_gpu(tile_dict, bucket_name, dataset_path,
                                         z_slice, n_blocks=3, pyramid_level=0, percentile_threshold=99):
    """
    GPU-accelerated calculation of accutance values for blocks within each tile across the entire grid.
    
    Args:
        tile_dict: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Z-slice to analyze
        n_blocks: Number of blocks per tile dimension (default 3)
        pyramid_level: Pyramid level to load
        percentile_threshold: Threshold for edge detection
        
    Returns:
        dict containing:
            grid_accutance: 2D array of accutance values
            coverage_map: Coverage map showing tile presence
            dimensions: Grid dimensions
            blocks_per_tile: Number of blocks per tile
    """
    
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")
    
    # First analyze the tile grid to get dimensions and positions
    grid_info = analyze_tile_grid(tile_dict, plot=False)
    grid_x, grid_y = grid_info['dimensions'][:2]
    coverage_map = grid_info['coverage_map']
    
    # Initialize the full accutance grid
    full_grid = np.full((grid_y * n_blocks, grid_x * n_blocks), np.nan)
    
    # Define Sobel kernels as PyTorch tensors
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    sobel_x = sobel_x.view(1, 1, 3, 3)  # Reshape for conv2d
    sobel_y = sobel_y.view(1, 1, 3, 3)  # Reshape for conv2d
    
    # Function to calculate accutance using PyTorch
    def calculate_accutance_torch(block_data, percentile):
        # Convert to PyTorch tensor and send to device
        if isinstance(block_data, np.ndarray):
            tensor = torch.from_numpy(block_data).float().to(device)
        else:
            tensor = block_data
            
        # Ensure tensor is 4D: [batch, channels, height, width]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
            
        # Normalize
        if tensor.max() > 0:
            tensor = tensor / tensor.max()
        
        # Apply Sobel filters
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        
        # Calculate gradient magnitude
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate threshold
        flat_grad = gradient_mag.view(-1)
        k = int(round(flat_grad.size(0) * (100 - percentile) / 100))
        threshold, _ = torch.kthvalue(flat_grad, k)
        
        # Create edge mask
        edge_mask = gradient_mag > threshold
        
        # Calculate mean accutance
        edge_values = gradient_mag[edge_mask]
        mean_accutance = edge_values.mean().item() if edge_values.size(0) > 0 else 0
        
        return mean_accutance

    # Process tiles in batches if possible
    batch_size = min(4, len(tile_dict))  # Adjust based on memory constraints
    tile_items = list(tile_dict.items())
    
    for batch_start in range(0, len(tile_items), batch_size):
        batch_end = min(batch_start + batch_size, len(tile_items))
        batch_tiles = tile_items[batch_start:batch_end]
        
        print(f"Processing batch of {len(batch_tiles)} tiles ({batch_start+1}-{batch_end} of {len(tile_items)})")
        
        # Load all tiles in this batch
        batch_data = []
        batch_positions = []
        
        for tile_id, tile_name in batch_tiles:
            print(f"Loading tile {tile_name}")
            # Extract tile position from name
            parts = tile_name.split('_')
            tile_x = int(parts[2])
            tile_y = int(parts[4])
            
            # Load tile data
            tile_data = load_tile_data(tile_name, bucket_name, dataset_path, pyramid_level)
            
            # Extract the z-slice
            if z_slice < tile_data.shape[2]:
                slice_data = tile_data[:, :, z_slice]
                batch_data.append(slice_data)
                batch_positions.append((tile_x, tile_y))
        
        # Skip if no valid data in this batch
        if not batch_data:
            continue
            
        # Process all tiles in this batch together
        for i, (slice_data, (tile_x, tile_y)) in enumerate(zip(batch_data, batch_positions)):
            # Convert the entire slice to a PyTorch tensor
            torch_slice = torch.from_numpy(slice_data).float().to(device)
            
            # Calculate block sizes
            y_block_size = slice_data.shape[0] // n_blocks
            x_block_size = slice_data.shape[1] // n_blocks
            
            # Process each block
            for i in range(n_blocks):
                for j in range(n_blocks):
                    y_start = i * y_block_size
                    y_end = (i + 1) * y_block_size
                    x_start = j * x_block_size
                    x_end = (j + 1) * x_block_size
                    
                    # Extract block data using PyTorch slicing
                    block_tensor = torch_slice[y_start:y_end, x_start:x_end]
                    
                    # Calculate accutance for this block
                    block_accutance = calculate_accutance_torch(block_tensor, percentile_threshold)
                    
                    # Place in the full grid
                    grid_y_pos = tile_y * n_blocks + i
                    grid_x_pos = tile_x * n_blocks + j
                    full_grid[grid_y_pos, grid_x_pos] = block_accutance
            
            # Free up GPU memory
            del torch_slice
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return {
        'grid_accutance': full_grid,
        'coverage_map': coverage_map,
        'dimensions': (grid_x, grid_y),
        'blocks_per_tile': n_blocks
    }