import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def measure_laplacian_variance(image, kernel_size=15, normalization='density'):
    """
    Calculate image quality using variance of the Laplacian, which works well for 
    spot-based images by measuring local contrast.
    
    Args:
        image: 2D numpy array
        kernel_size: Size of kernel for local normalization
        normalization: Method to normalize for feature density
        
    Returns:
        dict with quality metrics
    """
    from scipy import ndimage
    import cv2
    
    # Normalize image to [0,1] range
    img_norm = image.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Apply local contrast normalization to handle varying signal levels
    if kernel_size > 0:
        local_mean = ndimage.uniform_filter(img_norm, size=kernel_size)
        local_std = np.sqrt(ndimage.uniform_filter(img_norm**2, size=kernel_size) - local_mean**2)
        local_std = np.maximum(local_std, 0.001)  # Avoid division by zero
        img_norm = (img_norm - local_mean) / local_std
    
    # Calculate Laplacian - sensitive to rapid intensity changes (spots)
    laplacian = cv2.Laplacian(img_norm, cv2.CV_64F)
    
    # Calculate variance of Laplacian (measure of focus)
    lap_variance = np.var(laplacian)
    
    # Estimate feature density using thresholded image
    binary = img_norm > np.percentile(img_norm, 99)
    feature_density = np.sum(binary) / binary.size
    
    # Normalize based on feature density
    if normalization == 'density':
        # Higher is better, normalize by square root of density
        density_factor = max(0.01, np.sqrt(feature_density))
        normalized_variance = lap_variance / density_factor
    elif normalization == 'log_density':
        # Normalize by log of density
        density_factor = max(0.01, np.log10(1 + 100 * feature_density))
        normalized_variance = lap_variance / density_factor
    else:
        normalized_variance = lap_variance
    
    return {
        'laplacian_variance': lap_variance,
        'normalized_variance': normalized_variance,
        'feature_density': feature_density
    }


def measure_frequency_content(image, bin_cutoffs=[0.1, 0.25, 0.5, 0.75]):
    """
    Analyze image quality using frequency domain analysis.
    
    Args:
        image: 2D numpy array
        bin_cutoffs: Cutoff points for frequency bins as fraction of Nyquist frequency
        
    Returns:
        dict with spectral metrics
    """
    # Normalize image
    img_norm = image.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Apply windowing to reduce edge effects
    h, w = img_norm.shape
    y, x = np.ogrid[:h, :w]
    window = np.hanning(h)[:, np.newaxis] * np.hanning(w)
    img_windowed = img_norm * window
    
    # Compute FFT
    f_transform = np.fft.fft2(img_windowed)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    
    # Calculate radial profile (average energy at each frequency)
    cy, cx = h//2, w//2
    y_grid, x_grid = np.ogrid[:h, :w]
    r_grid = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    
    # Normalize radius to [0,1] where 1 is Nyquist frequency
    r_max = min(cx, cy)
    r_norm = r_grid / r_max
    
    # Create frequency bins
    bin_indices = []
    for i in range(len(bin_cutoffs)):
        if i == 0:
            indices = r_norm <= bin_cutoffs[i]
        else:
            indices = (r_norm > bin_cutoffs[i-1]) & (r_norm <= bin_cutoffs[i])
        bin_indices.append(indices)
    
    # Add the highest frequency bin
    bin_indices.append(r_norm > bin_cutoffs[-1])
    
    # Calculate energy in each bin
    total_energy = np.sum(magnitude_spectrum)
    bin_energies = [np.sum(magnitude_spectrum[indices]) / total_energy for indices in bin_indices]
    
    # Calculate high-to-low frequency ratio (measure of sharpness)
    high_freq_ratio = sum(bin_energies[2:]) / (sum(bin_energies[:2]) + 0.001)
    
    return {
        'frequency_bins': bin_energies,
        'high_to_low_ratio': high_freq_ratio,
        'total_energy': total_energy
    }


def measure_transcript_spot_quality(image, min_spot_size=3, max_spot_size=15, 
                                   percentile_threshold=99, normalization='density'):
    """
    Measure image quality specifically for transcriptomic data with RNA spots.
    
    Args:
        image: 2D numpy array
        min_spot_size, max_spot_size: Size range for valid RNA spots
        percentile_threshold: Threshold for spot detection
        normalization: Method to normalize for feature density
        
    Returns:
        dict with quality metrics
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    
    # Normalize image to [0,1]
    img_norm = image.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # 1. Detect spots using local maxima
    threshold = np.percentile(img_norm, percentile_threshold)
    binary_img = img_norm > threshold
    
    # Apply distance transform to separate nearby spots
    distance = ndimage.distance_transform_edt(binary_img)
    
    # Find local maxima (spots)
    coordinates = peak_local_max(distance, min_distance=min_spot_size//2)
    
    # If no spots found, return zeros
    if len(coordinates) == 0:
        return {
            'spot_count': 0,
            'mean_spot_contrast': 0,
            'mean_spot_snr': 0,
            'normalized_quality': 0,
            'spot_density': 0
        }
    
    # 2. Calculate spot properties
    spot_values = []
    spot_contrasts = []
    spot_snrs = []
    valid_spots = 0
    
    for coord in coordinates:
        y, x = coord
        
        # Extract local region around spot
        half_size = max_spot_size // 2
        y_min, y_max = max(0, y-half_size), min(img_norm.shape[0], y+half_size+1)
        x_min, x_max = max(0, x-half_size), min(img_norm.shape[1], x+half_size+1)
        
        local_region = img_norm[y_min:y_max, x_min:x_max]
        
        # Skip if region is too small
        if local_region.size <= 1:
            continue
            
        # Calculate spot intensity
        spot_intensity = img_norm[y, x]
        spot_values.append(spot_intensity)
        
        # Calculate local contrast (spot vs background)
        background = np.percentile(local_region, 25)  # Background estimation
        contrast = spot_intensity - background
        spot_contrasts.append(contrast)
        
        # Calculate local SNR
        bg_std = max(np.std(local_region), 0.001)  # Avoid division by zero
        snr = contrast / bg_std
        spot_snrs.append(snr)
        
        valid_spots += 1
    
    # 3. Calculate metrics
    spot_density = valid_spots / image.size
    mean_spot_intensity = np.mean(spot_values) if spot_values else 0
    mean_spot_contrast = np.mean(spot_contrasts) if spot_contrasts else 0
    mean_spot_snr = np.mean(spot_snrs) if spot_snrs else 0
    
    # 4. Normalize for feature density
    if normalization == 'density':
        # Higher is better, normalize by log of density to reduce impact of very dense regions
        density_factor = max(0.01, np.log10(1 + 100 * spot_density))
        normalized_quality = mean_spot_snr / density_factor
    elif normalization == 'log_count':
        # Normalize by log of count
        count_factor = max(1, np.log10(valid_spots + 1))
        normalized_quality = mean_spot_snr / count_factor
    else:
        normalized_quality = mean_spot_snr
    
    return {
        'spot_count': valid_spots,
        'mean_spot_intensity': mean_spot_intensity,
        'mean_spot_contrast': mean_spot_contrast, 
        'mean_spot_snr': mean_spot_snr,
        'normalized_quality': normalized_quality,
        'spot_density': spot_density
    }


def analyze_transcript_image_quality(image, z_slice=None):
    """
    Comprehensive analysis of transcriptomic image quality.
    
    Args:
        image: 3D or 2D numpy array
        z_slice: Z-slice to analyze (if 3D)
        
    Returns:
        dict with quality metrics
    """
    # Handle 2D/3D input
    if z_slice is not None and len(image.shape) == 3:
        img_slice = image[z_slice]
    else:
        img_slice = image
    
    # Apply all metrics
    spot_metrics = measure_transcript_spot_quality(img_slice)
    lap_metrics = measure_laplacian_variance(img_slice)
    freq_metrics = measure_frequency_content(img_slice)
    
    # Combine results
    results = {
        **spot_metrics,
        **lap_metrics,
        **freq_metrics
    }
    
    return results