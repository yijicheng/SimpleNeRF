# Shree KRISHNAya Namaha
# Estimates depth on RE10K scenes
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import datetime
import json
import time
import traceback
from pathlib import Path

from io import BytesIO
from PIL import Image

import torch
import numpy
import pandas
import simplejson
import skimage.io
from deepdiff import DeepDiff
from tqdm import tqdm
from collections import defaultdict
from tqdm import tqdm

import Tester01 as Tester

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming generation: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return

def convert_images(
    images,
): # -> Float[Tensor, "batch 3 height width"]:
    numpy_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        numpy_images.append(numpy.array(image))
    return numpy.stack(numpy_images)

def start_generation(split, chunk_s=None, chunk_e=None):
    # root_dirpath = Path('../../../') # root
    root_dirpath = Path('./data') # root
    database_dirpath = root_dirpath / "RealEstate10K/re10k" # data
    tmp_dirpath = root_dirpath / 'tmp' # tmp

    output_dirpath = database_dirpath / "estimated_depths/" / split # out
    output_dirpath.mkdir(parents=True, exist_ok=True)


    scene_to_chunk_json = database_dirpath / split / "index.json"
    scene_to_chunk = json.load(open(scene_to_chunk_json, "r"))

    chunk_to_scene = defaultdict(list)
    for scene, chunk in scene_to_chunk.items():
        chunk_to_scene[chunk].append(scene)

    tester = Tester.ColmapTester(tmp_dirpath)


    # for scene_num in tqdm(scene_nums):
    for chunk in tqdm(sorted(chunk_to_scene.keys())):
        chunk_block = torch.load(database_dirpath / split / chunk)
        depth_chunk = {}
        depth_chunk_path = output_dirpath / chunk
        if depth_chunk_path.exists():
            continue
        for scene in chunk_block:


            frames = convert_images(scene["images"]) # 0-255 # (98, 360, 640, 3)
            h, w = frames.shape[1], frames.shape[2]

            intrinsics = numpy.zeros((frames.shape[0], 3, 3))
            intrinsics[:, 2, 2] = 1
            intrinsics[:, 0, 0] = scene["cameras"][:, 0] * w # fx
            intrinsics[:, 1, 1] = scene["cameras"][:, 1] * h # fy
            intrinsics[:, 0, 2] = scene["cameras"][:, 2] * w # cx
            intrinsics[:, 1, 2] = scene["cameras"][:, 3] * h # cy


            w2c_mat = numpy.array(scene["cameras"][:, 6:]).reshape(-1, 3, 4)
            extrinsics = numpy.zeros((frames.shape[0], 4, 4))
            extrinsics[:, 3, 3] = 1
            extrinsics[:, :3, :] = w2c_mat
            

            depth_data_list, bounds_data = tester.estimate_sparse_depth(frames, extrinsics, intrinsics)
            if depth_data_list is None:
                continue
            
            depth_chunk[scene["key"]] = {"depth": depth_data_list, "bounds": bounds_data}

        torch.save(depth_chunk, depth_chunk_path)
        import pdb; pdb.set_trace()
            
    return


def demo1():
    """
    For a gen set
    :return:
    """

    start_generation("test")

    # gen_configs = {
    #     'generator': this_filename,
    #     'gen_num': 3,
    #     'gen_set_num': 3,
    #     'database_name': 'RealEstate10K',
    #     'database_dirpath': 'RealEstate10K/data',
    # }
    # start_generation(gen_configs)

    # gen_configs = {
    #     'generator': this_filename,
    #     'gen_num': 4,
    #     'gen_set_num': 4,
    #     'database_name': 'RealEstate10K',
    #     'database_dirpath': 'RealEstate10K/data',
    # }
    # start_generation(gen_configs)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    # try:
    main()
    run_result = 'Program completed successfully!'
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
    #     run_result = 'Error: ' + str(e)
    # end_time = time.time()
    # print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    # print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
