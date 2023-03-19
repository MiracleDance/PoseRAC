import pandas as pd
import numpy as np
import os
import cv2
from mediapipe.python.solutions import pose as mp_pose
import torch.onnx
import time
import yaml
import argparse
from utils import _annotation_transform, _generate_for_train, _generate_csv_label

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    old_time = time.time()

    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    root_dir = config['dataset']['dataset_root_dir']
    csv_label_path = config['dataset']['csv_label_path']

    print('start annotation transform')
    _annotation_transform(root_dir)

    print('start generate csv label')
    _generate_csv_label(root_dir, csv_label_path)

    print('start generate for train')
    _generate_for_train(root_dir)

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)
