import numpy as np
import zarr
import dask.array as da
import boto3
from pathlib import Path
import matplotlib.pyplot as plt

from ng_link import parsers
from collections import defaultdict

import stitch_utils
import seaborn as sns


def load_tile_data(tile_name: str, 
                  bucket_name: str, 
                  dataset_path: str, 
                  pyramid_level:int = 0,
                  dims_order:tuple = (1,2,0)) -> np.ndarray:
    """
    Load tile data from zarr
    (1,2,0) is the order of the dimensions in the zarr file
    """
    tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
    zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
    tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
    return tile_data.compute().transpose(dims_order)


def download_stitch_xmls(dataset_list,
                        save_dir=None,
                        overwrite=False):
    s3 = boto3.client("s3")
    bucket_name = "aind-open-data"
    xml_list = ["stitching_single_channel.xml", "stitching_spot_channels.xml"]

    xmls = {}
    for i, round_n in enumerate(dataset_list):
        round_dict = {}
        for xml in xml_list:
            xml_path = f"{round_n}/{xml}"
            round_dict[xml.split("_")[1]] = xml_path

            fn = save_dir / xml_path
            fn.parent.mkdir(parents=True, exist_ok=True)
            if fn.exists() and not overwrite:
                print(f"Skipping {xml_path} because it already exists")
                continue

            s3.download_file(
                Bucket=bucket_name,
                Key=xml_path,
                Filename=fn
            )
            print(f"Downloaded {xml_path} from S3")

        xmls[i+1] = round_dict

    return xmls

def get_thyme_xmls():
    r1 = "HCR_736963_2024-12-07_13-00-00"
    r2 = "HCR_736963_2024-12-13_13-00-00"
    r3 = "HCR_736963_2024-12-19_13-00-00"
    r4 = "HCR_736963_2025-01-09_13-00-00"
    r5 = "HCR_736963_2025-01-22_13-00-00"

    round_names = [r1, r2, r3, r4, r5]

    xmls = download_stitch_xmls(round_names, save_dir=Path(f'/home/matt.davis/code/hcr-stich/xml_data/'))
    return xmls


def map_channels_to_keys(tile_dict):
    """
    Create a mapping from channel names to lists of tile IDs.
    
    Args:
        tile_dict: Dictionary mapping IDs to tile names
                  Example: {0: 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr', ...}
    
    Returns:
        Dictionary mapping channel names to lists of tile IDs
        Example: {'405': [0, 1, 2, 3], '488': [4, 5, 6, 7]}
    """
    # Initialize dictionary to hold keys by channel
    channel_to_keys = {}
    
    # Process each tile
    for tile_id, tile_name in tile_dict.items():
        # Extract channel from tile name
        try:
            if '_ch_' in tile_name:
                channel = tile_name.split('_ch_')[1].split('.')[0]
            elif '.ch' in tile_name:
                channel = tile_name.split('.ch')[1].split('.')[0]
            else:
                parts = tile_name.split('_')
                for i, part in enumerate(parts):
                    if part == 'ch' and i+1 < len(parts):
                        channel = parts[i+1]
                        break
                else:
                    channel = 'unknown'
        except:
            channel = 'unknown'
        
        # Add key to channel list
        if channel not in channel_to_keys:
            channel_to_keys[channel] = []
        
        channel_to_keys[channel].append(tile_id)
    
    return channel_to_keys


def parse_bigstitcher_xml(xml_path):
    """
    Parse the XML file and return a dictionary of tile names, transforms, and other information.

    Works for both local and s3 paths.
    Works for both single and spot xml files from bigstitcher.
    """

    dataset_path = parsers.XmlParser.extract_dataset_path(xml_path=xml_path)
    # if start with /data, remove it (needed for s3 data)
    if dataset_path.startswith('/data/'):
        dataset_path = dataset_path[len('/data/'):]

    tile_names = parsers.XmlParser.extract_tile_paths(xml_path=xml_path)
    tile_transforms = parsers.XmlParser.extract_tile_transforms(xml_path=xml_path)
    tile_info = parsers.XmlParser.extract_info(xml_path=xml_path)
    net_transforms = calculate_net_transforms(tile_transforms)

    print(dataset_path)
    print(f"N tiles: {len(tile_names)}")


    channel_keys = map_channels_to_keys(tile_names)

    channels = list(channel_keys.keys())

    for channel in channels:
        print(f"{channel}: {len(channel_keys[channel])}")


    # put all in dict
    data = {
        "tile_names": tile_names,
        "tile_transforms": tile_transforms,
        "tile_info": tile_info,
        "net_transforms": net_transforms,
        "channel_keys_map": channel_keys,
        "channels": channels,
        "dataset_path": dataset_path
    }

    return data


def channel_data_from_parsed_xml(data, channel):
    """ArithmeticError

    Spot xml example:
    HCR_736963_2024-12-07_13-00-00/radial_correction.ome.zarr/
    N tiles: 278
    488: 68
    514: 68
    561: 68
    594: 68
    405: 6

    """

    channel_keys = data["channel_keys_map"]
    # assert channel in channel_keys
    assert channel in data["channels"]
    dataset_path = data["dataset_path"]
    tile_names = {key: data["tile_names"][key] for key in channel_keys[channel]}
    tile_transforms = {key: data["tile_transforms"][key] for key in channel_keys[channel]}
    #tile_info = {key: tile_info[key] for key in channel_keys[channel]}
    net_transforms = {key: data["net_transforms"][key] for key in channel_keys[channel]}

    channel_data = {
        "channel": channel,
        "channel_keys": channel_keys[channel],
        "tile_names": tile_names,
        "tile_transforms": tile_transforms,
        "net_transforms": net_transforms,
        "dataset_path": dataset_path
    }
    return channel_data


def load_tile_data(tile_name: str, 
                  bucket_name: str, 
                  dataset_path: str, 
                  pyramid_level:int = 0,
                  dims_order:tuple = (1,2,0)) -> np.ndarray:
    """
    Load tile data from zarr
    (1,2,0) is the order of the dimensions in the zarr file
    """
    tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
    zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
    tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
    return tile_data.compute().transpose(dims_order)


# def load_slice_data(tile_name: str, 
#                   bucket_name: str, 
#                   dataset_path: str, 
#                   slice_r
#                   pyramid_level:int = 0,
#                   dims_order:tuple = (1,2,0)) -> np.ndarray:
#     """
#     Load tile data from zarr
#     Zarrs are stored in (z,y,x) order
#     (1,2,0) is the order of the dimensions in the zarr file
#     """
#     tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
#     zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
#     tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
#     # get the slice at z_slice
#     slice_data = tile_data[z_slice, :, :]
#     return slice_data.compute().transpose(dims_order)


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

class TileData:
    """
    A class for lazily loading and manipulating tile data with flexible slicing and projection options.
    
    This class maintains the original dask array for memory efficiency and only computes data when needed.
    It provides methods to access data in different orientations (XY, ZY, ZX) and to perform projections.
    """
    
    def __init__(self, tile_name, bucket_name, dataset_path, pyramid_level=0):
        """
        Initialize the TileData object.
        
        Args:
            tile_name: Name of the tile
            bucket_name: S3 bucket name
            dataset_path: Path to dataset in bucket
            pyramid_level: Pyramid level to load (default 0)
        """
        self.tile_name = tile_name
        self.bucket_name = bucket_name
        self.dataset_path = dataset_path
        self.pyramid_level = pyramid_level
        self._data = None
        self._loaded = False
        
    def _load_lazy(self):
        """Lazily load the data as a dask array without computing"""
        if not self._loaded:
            tile_array_loc = f"{self.dataset_path}{self.tile_name}/{self.pyramid_level}"
            zarr_path = f"s3://{self.bucket_name}/{tile_array_loc}"
            self._data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
            self._loaded = True
            
            # Store shape information
            self.shape = self._data.shape
            # Assuming zarr is stored in (z,y,x) order
            self.z_dim, self.y_dim, self.x_dim = self.shape
    
    @property
    def data(self):
        """Get the full computed data in (x,y,z) order"""
        self._load_lazy()
        return self._data.compute().transpose(2,1,0)
    
    @property
    def data_raw(self):
        """Get the full computed data in original (z,y,x) order"""
        self._load_lazy()
        return self._data.compute()
    
    @property
    def dask_array(self):
        """Get the underlying dask array without computing"""
        self._load_lazy()
        return self._data

    def connect(self):
        """Establish connection to the data source without computing"""
        self._load_lazy()
        return self
    
    def get_slice(self, index, orientation='xy', compute=True):
        """
        Get a 2D slice through the data in the specified orientation.
        
        Args:
            index: Index of the slice
            orientation: One of 'xy', 'zy', 'zx' (default 'xy')
            compute: Whether to compute the dask array (default True)
            
        Returns:
            2D numpy array or dask array
        """
        self._load_lazy()
        
        if orientation == 'xy':
            # XY slice at specific Z
            if index >= self.z_dim:
                raise IndexError(f"Z index {index} out of bounds (max {self.z_dim-1})")
            slice_data = self._data[index, :, :]
        elif orientation == 'zy':
            # ZY slice at specific X
            if index >= self.x_dim:
                raise IndexError(f"X index {index} out of bounds (max {self.x_dim-1})")
            slice_data = self._data[:, :, index]
        elif orientation == 'zx':
            # ZX slice at specific Y
            if index >= self.y_dim:
                raise IndexError(f"Y index {index} out of bounds (max {self.y_dim-1})")
            slice_data = self._data[:, index, :]
        else:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'xy', 'zy', or 'zx'")
        
        if compute:
            return slice_data.compute()
        return slice_data
    
    def get_slice_range(self, start, end, axis='z', compute=True):
        """
        Get a range of slices along the specified axis.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            axis: One of 'z', 'y', 'x' (default 'z')
            compute: Whether to compute the dask array (default True)
            
        Returns:
            3D numpy array or dask array
        """
        self._load_lazy()
        
        if axis == 'z':
            if end > self.z_dim:
                raise IndexError(f"Z end index {end} out of bounds (max {self.z_dim})")
            slice_data = self._data[start:end, :, :]
        elif axis == 'y':
            if end > self.y_dim:
                raise IndexError(f"Y end index {end} out of bounds (max {self.y_dim})")
            slice_data = self._data[:, start:end, :]
        elif axis == 'x':
            if end > self.x_dim:
                raise IndexError(f"X end index {end} out of bounds (max {self.x_dim})")
            slice_data = self._data[:, :, start:end]
        else:
            raise ValueError(f"Unknown axis: {axis}. Use 'z', 'y', or 'x'")
        
        if compute:
            return slice_data.compute()
        return slice_data
    
    def project(self, axis='z', method='max', start=None, end=None, compute=True):
        """
        Project data along the specified axis using the specified method.
        
        Args:
            axis: One of 'z', 'y', 'x' (default 'z')
            method: One of 'max', 'mean', 'min', 'sum' (default 'max')
            start: Start index for projection range (default None = 0)
            end: End index for projection range (default None = full dimension)
            compute: Whether to compute the dask array (default True)
            
        Returns:
            2D numpy array or dask array
        """
        self._load_lazy()
        
        # Set default range
        if start is None:
            start = 0
        if end is None:
            if axis == 'z':
                end = self.z_dim
            elif axis == 'y':
                end = self.y_dim
            else:
                end = self.x_dim
        
        # Get the slice range
        range_data = self.get_slice_range(start, end, axis, compute=False)
        
        # Apply projection method
        if method == 'max':
            if axis == 'z':
                result = range_data.max(axis=0)
            elif axis == 'y':
                result = range_data.max(axis=1)
            else:  # axis == 'x'
                result = range_data.max(axis=2)
        elif method == 'mean':
            if axis == 'z':
                result = range_data.mean(axis=0)
            elif axis == 'y':
                result = range_data.mean(axis=1)
            else:  # axis == 'x'
                result = range_data.mean(axis=2)
        elif method == 'min':
            if axis == 'z':
                result = range_data.min(axis=0)
            elif axis == 'y':
                result = range_data.min(axis=1)
            else:  # axis == 'x'
                result = range_data.min(axis=2)
        elif method == 'sum':
            if axis == 'z':
                result = range_data.sum(axis=0)
            elif axis == 'y':
                result = range_data.sum(axis=1)
            else:  # axis == 'x'
                result = range_data.sum(axis=2)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'max', 'mean', 'min', or 'sum'")
        
        if compute:
            return result.compute()
        return result
    
    def get_orthogonal_views(self, z_index=None, y_index=None, x_index=None, compute=True):
        """
        Get orthogonal views (XY, ZY, ZX) at the specified indices.
        
        Args:
            z_index: Z index for XY view (default None = middle slice)
            y_index: Y index for ZX view (default None = middle slice)
            x_index: X index for ZY view (default None = middle slice)
            compute: Whether to compute the dask arrays (default True)
            
        Returns:
            dict with keys 'xy', 'zy', 'zx' containing the respective views
        """
        self._load_lazy()
        
        # Use middle slices by default
        if z_index is None:
            z_index = self.z_dim // 2
        if y_index is None:
            y_index = self.y_dim // 2
        if x_index is None:
            x_index = self.x_dim // 2
        
        # Get the three orthogonal views
        xy_view = self.get_slice(z_index, 'xy', compute)
        zy_view = self.get_slice(x_index, 'zy', compute)
        zx_view = self.get_slice(y_index, 'zx', compute)
        
        return {
            'xy': xy_view,
            'zy': zy_view,
            'zx': zx_view
        }

    def set_pyramid_level(self, level: int):
        """
        Set the pyramid level and clear any loaded data.
        
        Args:
            level: New pyramid level to use
            
        Returns:
            self (for method chaining)
        """
        if level != self.pyramid_level:
            self.pyramid_level = level
            # Clear loaded data so it will be reloaded at new pyramid level
            self._data = None
            self._loaded = False
        return self


    def calculate_max_slice(self, level_to_use=2):
        """

        Use pyramidal level 3 and calulate the mean of the slices in all 3 dimensions,
        report back using the index for all pyramid levels.

        scale = int(2**pyramid_level)

        Help to get estimates of where lots of signal is in the tile.

        """
        level_to_use = level_to_use
        self.set_pyramid_level(level_to_use)

        # first load the data
        data = self.data

        max_slices = {}
        # find index of max slice in z
        max_slice_z = data.mean(axis=0)
        max_slice_z_index = np.unravel_index(max_slice_z.argmax(), max_slice_z.shape)
        max_slice_y = data.mean(axis=1)
        max_slice_y_index = np.unravel_index(max_slice_y.argmax(), max_slice_y.shape)
        max_slice_x = data.mean(axis=2)
        max_slice_x_index = np.unravel_index(max_slice_x.argmax(), max_slice_x.shape)

        pyramid_levels = [0,1,2,3]

        max_slices[level_to_use] = {
            "z": int(max_slice_z_index[0]),
            "y": int(max_slice_y_index[0]),
            "x": int(max_slice_x_index[0])
        }

        # remove level_to_use from pyramid_levels
        pyramid_levels.remove(level_to_use)

        for level in pyramid_levels:
            if level_to_use >= level:
                scale_factor = 2**(level_to_use - level)
            else:
                print(f"level_to_use: {level_to_use}, level: {level}")
                scale_factor = 1/(2**(level - level_to_use))
            max_slices[level] = {
                "z": int(max_slice_z_index[0] * scale_factor),
                "y": int(max_slice_y_index[0] * scale_factor),
                "x": int(max_slice_x_index[0] * scale_factor)
            }

        # sort keys by int value
        max_slices = dict(sorted(max_slices.items(), key=lambda item: int(item[0])))

        return max_slices


class PairedTiles:
    """
    Class to hold a pair of adjacent tiles and visualize their overlap in 3D.
    
    This class handles translation-only registration between tiles and provides 
    methods to visualize slices through the composite volume in any orientation.
    """
    
    def __init__(self, tile1, tile2, transform1, transform2, names=None, clip_percentiles=(1, 99)):
        """
        Initialize with two TileData objects and their transformation matrices.
        
        Args:
            tile1: First TileData object
            tile2: Second TileData object
            transform1: 4x4 transformation matrix for tile1
            transform2: 4x4 transformation matrix for tile2
            names: Optional tuple of (name1, name2) for the tiles
            clip_percentiles: Tuple of (min_percentile, max_percentile) for intensity clipping
        """
        self.tile1 = tile1
        self.tile2 = tile2
        self.transform1 = transform1.copy()
        self.transform2 = transform2.copy()
        
        self.pyramid_level1 = tile1.pyramid_level
        self.pyramid_level2 = tile2.pyramid_level
        
        self.shape1 = tile1.shape
        self.shape2 = tile2.shape
        
        # Store tile names
        if names is None:
            self.name1, self.channel1 = tile1.tile_name
            self.name2, self.channel2 = tile2.tile_name
        else:
            self.name1, self.name2 = names

        self.clip_percentiles = clip_percentiles
        
        self._scale_transforms()
        self._calculate_bounds()
        self.load_data()
    
    def _scale_transforms(self):
        """Scale the translation components of transforms based on pyramid level."""
        scale_factor1 = 2**self.pyramid_level1
        self.scaled_transform1 = self.transform1.copy()
        self.scaled_transform1[:3, 3] = self.scaled_transform1[:3, 3] / scale_factor1
        
        scale_factor2 = 2**self.pyramid_level2
        self.scaled_transform2 = self.transform2.copy()
        self.scaled_transform2[:3, 3] = self.scaled_transform2[:3, 3] / scale_factor2
        
        self.scale_factor1 = scale_factor1
        self.scale_factor2 = scale_factor2
    
    def _calculate_bounds(self):
        """Calculate the global bounds for the composite volume."""
        # For translation-only transforms, we need to find:
        # 1. The minimum coordinates (for the origin of the composite array)
        # 2. The maximum coordinates (to determine the size of the composite array)
        
        # Get corners of tile1 in global space
        shape1_zyx = np.array(self.shape1)  # (z, y, x)
        corners1 = np.array([
            [0, 0, 0],  # origin
            [shape1_zyx[0], 0, 0],  # max z
            [0, shape1_zyx[1], 0],  # max y
            [0, 0, shape1_zyx[2]],  # max x
            [shape1_zyx[0], shape1_zyx[1], 0],  # max z, y
            [shape1_zyx[0], 0, shape1_zyx[2]],  # max z, x
            [0, shape1_zyx[1], shape1_zyx[2]],  # max y, x
            [shape1_zyx[0], shape1_zyx[1], shape1_zyx[2]]  # max z, y, x
        ])
        
        # Transform corners to global space (for translation only)
        global_corners1 = corners1 + self.scaled_transform1[:3, 3]
        
        # Repeat for tile2
        shape2_zyx = np.array(self.shape2)
        corners2 = np.array([
            [0, 0, 0],
            [shape2_zyx[0], 0, 0],
            [0, shape2_zyx[1], 0],
            [0, 0, shape2_zyx[2]],
            [shape2_zyx[0], shape2_zyx[1], 0],
            [shape2_zyx[0], 0, shape2_zyx[2]],
            [0, shape2_zyx[1], shape2_zyx[2]],
            [shape2_zyx[0], shape2_zyx[1], shape2_zyx[2]]
        ])
        
        global_corners2 = corners2 + self.scaled_transform2[:3, 3]
        
        # Combine all corners and find min/max
        all_corners = np.vstack([global_corners1, global_corners2])
        self.min_corner = np.floor(np.min(all_corners, axis=0)).astype(int)
        self.max_corner = np.ceil(np.max(all_corners, axis=0)).astype(int)
        
        # Calculate composite shape
        self.composite_shape = self.max_corner - self.min_corner
        
        # Calculate offsets for each tile in the composite array
        self.offset1 = (self.scaled_transform1[:3, 3] - self.min_corner).astype(int)
        self.offset2 = (self.scaled_transform2[:3, 3] - self.min_corner).astype(int)
        
        # Print some debug info
        print(f"Composite shape: {self.composite_shape}")
        print(f"Tile1 offset: {self.offset1}")
        print(f"Tile2 offset: {self.offset2}")
    
    def load_data(self):
        """Load and transform tile data into composite space with percentile clipping."""
        composite_shape = tuple(self.composite_shape) + (3,)
        self.composite = np.zeros(composite_shape, dtype=np.float32)
        
        
        
        data1 = self.tile1.data.copy() 
        data2 = self.tile2.data.copy()
        
        print(f"Tile1 shape: {data1.shape}, non-zero pixels: {np.count_nonzero(data1)}")
        print(f"Tile2 shape: {data2.shape}, non-zero pixels: {np.count_nonzero(data2)}")
        
        min_percentile, max_percentile = self.clip_percentiles
        
        # Clip and normalize tile1 data
        if np.any(data1 > 0):
            # non zero for min
            p_min1 = np.percentile(data1[data1 > 0], min_percentile)
            p_max1 = np.percentile(data1[data1 > 0], max_percentile)
            print(f"Tile1 percentiles: {min_percentile}% = {p_min1}, {max_percentile}% = {p_max1}")
            
            data1_clipped = np.clip(data1, p_min1, p_max1)
            
            # normalize to [0, 1]
            data1_norm = (data1_clipped - p_min1) / (p_max1 - p_min1) if p_max1 > p_min1 else np.zeros_like(data1_clipped)
            print(f"Tile1 normalized range: {data1_norm.min()} to {data1_norm.max()}")
        else:
            data1_norm = np.zeros_like(data1, dtype=np.float32)
            p_min1, p_max1 = 0, 0
        
        # Clip and normalize tile2 data
        if np.any(data2 > 0):
            p_min2 = np.percentile(data2[data2 > 0], min_percentile)
            p_max2 = np.percentile(data2[data2 > 0], max_percentile)
            print(f"Tile2 percentiles: {min_percentile}% = {p_min2}, {max_percentile}% = {p_max2}")
            
            data2_clipped = np.clip(data2, p_min2, p_max2)
            
            # Normalize to [0, 1]
            data2_norm = (data2_clipped - p_min2) / (p_max2 - p_min2) if p_max2 > p_min2 else np.zeros_like(data2_clipped)
            print(f"Tile2 normalized range: {data2_norm.min()} to {data2_norm.max()}")
        else:
            data2_norm = np.zeros_like(data2, dtype=np.float32)
            p_min2, p_max2 = 0, 0
        
        self.percentile_values = {
            'tile1': (p_min1, p_max1),
            'tile2': (p_min2, p_max2)
        }
        
        # put data into composite
        z1, y1, x1 = data1.shape
        oz1, oy1, ox1 = self.offset1
        
        print(f"Tile1 offset in composite: {self.offset1}")
        print(f"Tile2 offset in composite: {self.offset2}")
        
        # Calculate the actual space available in the composite array
        z1_space = min(z1, self.composite_shape[0] - oz1)
        y1_space = min(y1, self.composite_shape[1] - oy1)
        x1_space = min(x1, self.composite_shape[2] - ox1)
        
        if z1_space < z1 or y1_space < y1 or x1_space < x1:
            print(f"Warning: Tile1 extends beyond composite bounds. Clipping tile data.")
            print(f"Available space: {z1_space}, {y1_space}, {x1_space}")
        
        # Place tile1 data, clipping if necessary
        self.composite[oz1:oz1+z1_space, oy1:oy1+y1_space, ox1:ox1+x1_space, 0] = \
            data1_norm[:z1_space, :y1_space, :x1_space]
        
        # Tile2 goes into green channel
        z2, y2, x2 = data2.shape
        oz2, oy2, ox2 = self.offset2
        
        # Calculate the actual space available in the composite array
        z2_space = min(z2, self.composite_shape[0] - oz2)
        y2_space = min(y2, self.composite_shape[1] - oy2)
        x2_space = min(x2, self.composite_shape[2] - ox2)
        
        if z2_space < z2 or y2_space < y2 or x2_space < x2:
            print(f"Warning: Tile2 extends beyond composite bounds. Clipping tile data.")
            print(f"Available space: {z2_space}, {y2_space}, {x2_space}")
        
        # Place tile2 data, clipping if necessary
        self.composite[oz2:oz2+z2_space, oy2:oy2+y2_space, ox2:ox2+x2_space, 1] = \
            data2_norm[:z2_space, :y2_space, :x2_space]

        # print(f"Composite shape: {self.composite_shape}")
        # # Rotate composite 90 degrees if height < width
        # if self.composite_shape[1] < self.composite_shape[0]:
        #     self.composite = np.rot90(self.composite, k=-1, axes=(0,1))
        #     self.should_rotate = True
        # else:
        #     self.should_rotate = False
        # Create overlap mask (where both tiles have data)
        self.overlap_mask = (self.composite[..., 0] > 0) & (self.composite[..., 1] > 0)
        # if self.should_rotate:
        #     self.overlap_mask = np.rot90(self.overlap_mask, k=-1, axes=(0,1))
        
        overlap_volume = np.sum(self.overlap_mask)
        total_volume = np.prod(self.composite_shape)
        print(f"Overlap volume: {overlap_volume} voxels ({overlap_volume/total_volume:.2%} of composite)")
        
        print(f"Red channel (Tile1) max value in composite: {self.composite[..., 0].max()}")
        print(f"Green channel (Tile2) max value in composite: {self.composite[..., 1].max()}")
    
    def get_slice(self, index, orientation='xy', overlap_only=False, padding=20):
        """
        Get a slice from the composite volume.
        
        Args:
            index: Index along the slicing dimension
            orientation: One of 'xy', 'zy', 'zx'
            overlap_only: If True, show only the overlap region
            padding: Number of pixels to pad around overlap region
            
        Returns:
            RGB slice data
        """
        if orientation == 'xy':
            if index >= self.composite_shape[2]:
                raise IndexError(f"Z index {index} out of bounds (max {self.composite_shape[2]-1})")
            slice_data = self.composite[:, :, index, :]
            slice_mask = self.overlap_mask[:, :, index] if overlap_only else None
            
        elif orientation == 'zy':
            if index >= self.composite_shape[0]:
                raise IndexError(f"X index {index} out of bounds (max {self.composite_shape[0]-1})")
            slice_data = self.composite[index, :, :, :]
            slice_mask = self.overlap_mask[index, :, :] if overlap_only else None
            
        elif orientation == 'zx':
            if index >= self.composite_shape[1]:
                raise IndexError(f"Y index {index} out of bounds (max {self.composite_shape[1]-1})")
            slice_data = self.composite[:, index, :, :]
            slice_mask = self.overlap_mask[:, index, :] if overlap_only else None
        
        else:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'xy', 'zy', or 'zx'")
        
        # If overlap_only is True, find the overlap region and add padding
        if overlap_only and slice_mask is not None:
            # Find the bounds of overlap region
            rows, cols = np.where(slice_mask)
            if len(rows) > 0 and len(cols) > 0:
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
                
                # Add padding
                rmin = max(0, rmin - padding)
                rmax = min(slice_data.shape[0], rmax + padding)
                cmin = max(0, cmin - padding)
                cmax = min(slice_data.shape[1], cmax + padding)
                
                # Crop to padded overlap region
                slice_data = slice_data[rmin:rmax+1, cmin:cmax+1]
            else:
                # No overlap found
                slice_data = np.zeros((1, 1, 3))
            
        return slice_data

    def visualize_slice(self, index, orientation='xy', overlap_only=False, ax=None, padding=20, rotate_z=True):
        """
        Visualize a slice from the composite volume.
        Shows the overlap region with padding if overlap_only is True.
        
        Args:
            index: Index along the slicing dimension
            orientation: One of 'xy', 'zy', 'zx' (default 'xy')
            overlap_only: If True, show only the overlap region
            ax: Matplotlib axis to plot on
            padding: Number of pixels to pad around overlap region
            rotate_z: If True, display Z as the vertical axis in XZ and YZ views
        """
        slice_data = self.get_slice(index, orientation, overlap_only, padding=padding)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Rotate the slice if needed
        if rotate_z and orientation in ['zx', 'zy']:
            slice_data = np.rot90(slice_data,3)
        
        ax.imshow(slice_data)
        
        # Adjust axis labels based on rotation
        if rotate_z:
            axis_labels = {
                'xy': ('Y', 'X', 'Z'),  # XY view unchanged
                'zy': ('Z', 'Y', 'X'),  # YZ view becomes ZY
                'zx': ('Z', 'X', 'Y')   # XZ view becomes ZX
            }
        else:
            axis_labels = {
                'xy': ('Y', 'X', 'Z'),
                'zy': ('Y', 'Z', 'X'),
                'zx': ('X', 'Z', 'Y')
            }
        
        ylabel, xlabel, slice_dim = axis_labels[orientation]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{orientation.upper()} slice at {slice_dim}={index}")
        
        return fig, ax

    def visualize_orthogonal_views(self, z_slice=None, y_slice=None, x_slice=None, 
                                 overlap_only=False, padding=20, rotate_z=True):
        """
        Visualize orthogonal views of the composite volume.
        
        Args:
            z_slice, y_slice, x_slice: Slice indices
            overlap_only: If True, show only the overlap region
            padding: Number of pixels to pad around overlap region
            rotate_z: If True, display Z as the vertical axis in XZ and YZ views
        """
        # Use middle slices by default
        if x_slice is None:
            x_slice = self.composite_shape[0] // 2
        if y_slice is None:
            y_slice = self.composite_shape[1] // 2 
        if z_slice is None:
            z_slice = self.composite_shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot views with padding
        self.visualize_slice(z_slice, 'xy', overlap_only, axes[0], padding=padding, rotate_z=rotate_z)
        self.visualize_slice(y_slice, 'zx', overlap_only, axes[1], padding=padding, rotate_z=rotate_z)
        self.visualize_slice(x_slice, 'zy', overlap_only, axes[2], padding=padding, rotate_z=rotate_z)
        
        plt.suptitle(f"Orthogonal Views of Paired Tiles\n{self.name1} (red) and {self.name2} (green)", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig, axes


def visualize_multichannel_paired_tiles(tile1_name, tile2_name, data, 
                                      channels=['405', '488', '514', '561', '594', '638'],
                                      pyramid_level=2, overlap_only=False, padding='auto', rotate_z=True):
    """
    Visualize orthogonal views for all channels of a pair of tiles.
    
    Args:
        tile1_name: Name of first tile
        tile2_name: Name of second tile
        data: Parsed XML data
        channels: List of channels to visualize
        pyramid_level: Pyramid level to load
        overlap_only: Whether to show only overlap regions
        padding: Padding around overlap regions
        rotate_z: Whether to rotate Z axis to vertical
    """
    sns.set_context("talk")
    # Create figure with n_channels rows and 3 columns
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 3, figsize=(18, 10*n_channels))
    
    # Get tile IDs
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]

    
    parsed_name1, ch1 = stitch_utils.parse_tile_name(tile1_name)
    parsed_name2, ch2 = stitch_utils.parse_tile_name(tile2_name)


    if padding == 'auto':
        padding = 16 * 2**(3-pyramid_level)
    
    # Function to replace channel in tile name
    def replace_channel(tile_name, new_channel):
        parts = tile_name.split('_ch_')
        return f"{parts[0]}_ch_{new_channel}.zarr"
    
    # Process each channel
    for i, channel in enumerate(channels):
        # Replace channel in tile names
        ch_tile1 = replace_channel(tile1_name, channel)
        ch_tile2 = replace_channel(tile2_name, channel)

        print(f"\nChannel {channel}\n----------")
        print(f"Tile1: {ch_tile1}")
        print(f"Tile2: {ch_tile2}")
        
        try:
            # Create paired tiles for this channel
            paired_tiles = PairedTiles(
                tile1=TileData(ch_tile1, "aind-open-data", data["dataset_path"], pyramid_level).connect(),
                tile2=TileData(ch_tile2, "aind-open-data", data["dataset_path"], pyramid_level).connect(),
                transform1=transform1,
                transform2=transform2,
                names=(f"{ch_tile1} ({channel})", f"{ch_tile2} ({channel})")
            )
            
            # Load the data
            paired_tiles.load_data()
            
            # Get middle slices
            z_slice = paired_tiles.composite_shape[2] // 2
            y_slice = paired_tiles.composite_shape[1] // 2
            x_slice = paired_tiles.composite_shape[0] // 2
            
            # Plot the three views
            paired_tiles.visualize_slice(z_slice, 'xy', overlap_only, axes[i,0], padding=padding, rotate_z=rotate_z)
            paired_tiles.visualize_slice(y_slice, 'zx', overlap_only, axes[i,1], padding=padding, rotate_z=rotate_z)
            paired_tiles.visualize_slice(x_slice, 'zy', overlap_only, axes[i,2], padding=padding, rotate_z=rotate_z)
            
            # Make titles more compact
            axes[i,0].set_title(f"XY@Z={z_slice}", pad=2)
            
            # Add clims above middle plot
            red_min, red_max = paired_tiles.percentile_values['tile1']
            green_min, green_max = paired_tiles.percentile_values['tile2']
            clim_text = f"Ch {channel}\nRed min/max: {int(red_min)}-{int(red_max)}\nGreen min/max: {int(green_min)}-{int(green_max)}"
            axes[i,1].set_title(f"{clim_text}\nZX@Y={y_slice}", pad=2)
            
            axes[i,2].set_title(f"ZY@X={x_slice}", pad=2)
            # add channel name to left of XY plot
            

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            # show traceback
            # Create empty plots for this row
            # for j in range(3):
            #     axes[i,j].imshow(np.zeros((100,100,3)))
            #     axes[i,j].set_title(f"Channel {channel} - Error")
    
    #plt.suptitle(f"Orthogonal Views of Paired Tiles Across Channels\n{data['dataset_path']}\n{parsed_name1} and {parsed_name2}", fontsize=20)
    plt.suptitle(f"{data['dataset_path']}\n{parsed_name1} and {parsed_name2}", fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig, axes




def visualize_paired_tiles(tile1_name, tile2_name, data, pyramid_level=1, 
                           bucket_name='aind-open-data', overlap_only=False, padding='auto'):

    # 20 is good for pyramid level 3, scale up for lower levels by factor of 4
    if padding == 'auto':
        padding = 16 * 2**(3-pyramid_level)
    # Create TileData objects
    tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()
    tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()

    # Get tile IDs
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]
    
    # Parse tile names for display
    parsed_name1 = stitch_utils.parse_tile_name(tile1_name)
    parsed_name2 = stitch_utils.parse_tile_name(tile2_name)
    
    # Create PairedTiles object
    paired = PairedTiles(tile1, tile2, transform1, transform2, names=(parsed_name1, parsed_name2))
    
    # Visualize orthogonal views
    fig, axes = paired.visualize_orthogonal_views(overlap_only=overlap_only, padding=padding)
    
    return paired, fig, axes

def get_net_transforms(data, tile_name):
    """
    Get the net transforms for a pair of tiles from the data dictionary.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing dataset information including tile_names and net_transforms
    tile_name : str
        Name of the first tile

        
    Returns:
    --------
    transform - The net transform for the tile
    """
    # Get tile IDs
    tile_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile_name)]

    # Get transforms
    transform = data["net_transforms"][tile_id]
    
    return transform

def create_paired_tiles(data, tile1_name, tile2_name, bucket_name, pyramid_level=2):
    """
    Create a PairedTiles object from two tile names and a data dictionary.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing dataset information including tile_names, net_transforms, and dataset_path
        From BigSticher parsed XML
    tile1_name : str
        Name of the first tile
    tile2_name : str
        Name of the second tile
    bucket_name : str
        Name of the S3 bucket
    pyramid_level : int, optional
        Pyramid level to use (default: 0)
        
    Returns:
    --------
    paired : io_utils.PairedTiles
        PairedTiles object containing the two connected tiles with transforms
    """
    # Create TileData objects
    tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()
    tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()

    # Get tile IDs and transforms
    transform1 = get_net_transforms(data, tile1_name)
    transform2 = get_net_transforms(data, tile2_name)
    # Parse tile names for display
    parsed_name1, ch1 = stitch_utils.parse_tile_name(tile1_name)
    parsed_name2, ch2 = stitch_utils.parse_tile_name(tile2_name)

    # Create PairedTiles object
    paired = PairedTiles(tile1, tile2, transform1, transform2, names=(parsed_name1, parsed_name2))
    
    return paired



def figure_tile_overlap_4_slices(tile1_name, tile2_name, data, pyramid_level=1, bucket_name='aind-open-data'):
# Create TileData objects
    tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()
    tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()


    # look up the values in data["tile_names"] to get the ids (which is the key)
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms for the tiles
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]

    n_cols = 4
    size = 10
    fig, axes = plt.subplots(1, n_cols, figsize=(size,size), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    # Calculate z-slices at 20%, 40%, 60%, and 80% through the z dimension
    z_min = max(0, min(tile1.shape[0], tile2.shape[0]))
    z_slices = [int(z_min * p) for p in [0.2, 0.4, 0.6, 0.8]]

    for i, z_slice in enumerate(z_slices):
        result = stitch_utils.visualize_tile_overlap(tile1, tile2, transform1, transform2, 
                                                    z_slice=z_slice, padding=50) # 1=50, 2 = 30, 3 = 20
        
        # Check if overlap is longer in x than y and transpose if needed
        composite = result['composite']
        overlap_shape = composite.shape
        if overlap_shape[1] > overlap_shape[0]:  # if width > height
            # transpose, but leave 3 dims
            composite = composite.transpose(1, 0, 2)
        
        axes[i].imshow(composite)
        axes[i].set_title(f'Z={z_slice}')
        axes[i].axis('on')
    tile1_name, ch1 = stitch_utils.parse_tile_name(tile1.tile_name)
    tile2_name, ch2 = stitch_utils.parse_tile_name(tile2.tile_name)
    # add whitespace above title
    plt.suptitle(f'Tile Overlap\nRed={tile1_name}, Green={tile2_name} Ch={ch1}', y=0.85)
    plt.tight_layout()
    #plt.subplots_adjust(top=0.8)
    plt.show()


