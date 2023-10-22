# Shree KRISHNAya Namaha
# RMSE measure between predicted frames and ground truth frames
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skimage.transform
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-10]


class RMSE:
    def __init__(self, frames_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.frames_data = frames_data
        self.verbose_log = verbose_log
        return

    @staticmethod
    def compute_frame_rmse(gt_frame: numpy.ndarray, eval_frame: numpy.ndarray):
        gt_frame = gt_frame
        eval_frame = eval_frame
        error = gt_frame.astype('float') - eval_frame.astype('float')
        rmse = numpy.sqrt(numpy.mean(numpy.square(error)))
        return rmse

    def compute_avg_rmse(self, old_data: pandas.DataFrame, database_dirpath: Path, pred_videos_dirpath: Path,
                         pred_frames_dirname: str, resolution_suffix: str, downsampling_factor: int):
        """

        :param old_data:
        :param database_dirpath: Should be path to databases/NeRF_LLFF/data
        :param pred_videos_dirpath:
        :param pred_frames_dirname:
        :param resolution_suffix:
        :param downsampling_factor:
        :return:
        """
        qa_scores = []
        for i, frame_data in tqdm(self.frames_data.iterrows(), total=self.frames_data.shape[0], leave=self.verbose_log):
            scene_name, pred_frame_num = frame_data
            if old_data is not None and old_data.loc[
                (old_data['scene_name'] == scene_name) & (old_data['pred_frame_num'] == pred_frame_num)
            ].size > 0:
                continue
            gt_frame_path = database_dirpath / f'all/database_data/{scene_name}/rgb{resolution_suffix}/{pred_frame_num:04}.png'
            pred_frame_path = pred_videos_dirpath / f'{scene_name}/{pred_frames_dirname}/{pred_frame_num:04}.png'
            if not pred_frame_path.exists():
                continue
            gt_frame = self.read_image(gt_frame_path)
            if downsampling_factor > 1:
                gt_frame = self.downsample_image(gt_frame, downsampling_factor)
            pred_frame = self.read_image(pred_frame_path)
            qa_score = self.compute_frame_rmse(gt_frame, pred_frame)
            qa_scores.append([scene_name, pred_frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['scene_name', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        merged_data = merged_data.round({this_metric_name: 4, })

        avg_rmse = numpy.mean(merged_data[this_metric_name])
        if isinstance(avg_rmse, numpy.ndarray):
            avg_rmse = avg_rmse.item()
        avg_rmse = numpy.round(avg_rmse, 4)
        return avg_rmse, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['scene_name', 'pred_frame_num'], inplace=True)
            new_data.set_index(['scene_name', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def downsample_image(image: numpy.ndarray, downsampling_factor: int):
        downsampled_image = skimage.transform.rescale(image, scale=1 / downsampling_factor, preserve_range=True,
                                                      multichannel=True, anti_aliasing=True)
        downsampled_image = numpy.round(downsampled_image).astype('uint8')
        return downsampled_image


# noinspection PyUnusedLocal
def start_qa(pred_videos_dirpath: Path, database_dirpath: Path, frames_datapath: Path, pred_frames_dirname: str,
             resolution_suffix, downsampling_factor: int):
    if not pred_videos_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: pred_videos_dirpath does not exist')
        return

    if not database_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: database_dirpath does not exist')
        return

    qa_scores_filepath = pred_videos_dirpath / 'QA_Scores.json'
    rmse_data_path = pred_videos_dirpath / f'QA_Scores/{pred_frames_dirname}/{this_metric_name}_FrameWise.csv'
    if qa_scores_filepath.exists():
        with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
            qa_scores = json.load(qa_scores_file)
    else:
        qa_scores = {}

    if pred_frames_dirname in qa_scores:
        if this_metric_name in qa_scores[pred_frames_dirname]:
            avg_rmse = qa_scores[pred_frames_dirname][this_metric_name]
            print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_frames_dirname}: {avg_rmse}')
            print('Running QA again.')
    else:
        qa_scores[pred_frames_dirname] = {}

    if rmse_data_path.exists():
        rmse_data = pandas.read_csv(rmse_data_path)
    else:
        rmse_data = None

    frames_data = pandas.read_csv(frames_datapath)[['scene_name', 'pred_frame_num']]

    mse_computer = RMSE(frames_data)
    avg_rmse, rmse_data = mse_computer.compute_avg_rmse(rmse_data, database_dirpath, pred_videos_dirpath,
                                                        pred_frames_dirname, resolution_suffix, downsampling_factor)
    if numpy.isfinite(avg_rmse):
        qa_scores[pred_frames_dirname][this_metric_name] = avg_rmse
        print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_frames_dirname}: {avg_rmse}')
        with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
            simplejson.dump(qa_scores, qa_scores_file, indent=4)
        rmse_data_path.parent.mkdir(parents=True, exist_ok=True)
        rmse_data.to_csv(rmse_data_path, index=False)
    return avg_rmse


def demo1():
    root_dirpath = Path('../../../../')
    pred_videos_dirpath = root_dirpath / 'runs/testing/test1011'
    database_dirpath = root_dirpath / 'data/databases/NeRF_LLFF/data'
    frames_data_path = database_dirpath / 'train_test_sets/set02/TestVideosData.csv'
    pred_frames_dirname = 'predicted_frames'
    resolution_suffix = '_down4'
    downsampling_factor = 1
    avg_rmse = start_qa(pred_videos_dirpath, database_dirpath, frames_data_path, pred_frames_dirname, resolution_suffix,
                        downsampling_factor)
    return avg_rmse


def demo2(args: dict):
    pred_videos_dirpath = args['pred_videos_dirpath']
    if pred_videos_dirpath is None:
        raise RuntimeError(f'Please provide pred_videos_dirpath')
    pred_videos_dirpath = Path(pred_videos_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    frames_datapath = args['frames_datapath']
    if frames_datapath is None:
        raise RuntimeError(f'Please provide frames_datapath')
    frames_datapath = Path(frames_datapath)

    pred_frames_dirname = args['pred_frames_dirname']
    resolution_suffix = args['resolution_suffix']
    downsampling_factor = args['downsampling_factor']

    avg_rmse = start_qa(pred_videos_dirpath, database_dirpath, frames_datapath, pred_frames_dirname, resolution_suffix,
                        downsampling_factor)
    return avg_rmse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_frames_dirname', default='predicted_frames')
    parser.add_argument('--resolution_suffix', default='_down4')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_videos_dirpath': args.pred_videos_dirpath,
        'database_dirpath': args.database_dirpath,
        'frames_datapath': args.frames_datapath,
        'pred_frames_dirname': args.pred_frames_dirname,
        'resolution_suffix': args.resolution_suffix,
        'downsampling_factor': args.downsampling_factor,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        avg_rmse = demo1()
    elif args['demo_function_name'] == 'demo2':
        avg_rmse = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return avg_rmse


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    args = parse_args()
    try:
        output_score = main(args)
        run_result = f'Program completed successfully!\nAverage {this_metric_name}: {output_score}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
