import numpy as np
import zarr
import dask.array as da
import boto3
from pathlib import Path

from ng_link import parsers



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
    bucket_name = "aind-opne-data"
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