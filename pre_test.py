import pandas as pd
import numpy as np
import os
import cv2
from mediapipe.python.solutions import pose as mp_pose
import torch.onnx
import time
import yaml
import argparse


torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:,:,0], axis = 1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:,:,0], axis = 1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:,:,1], axis = 1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:,:,1], axis = 1), 1)

    z_max = np.expand_dims(np.max(all_landmarks[:,:,2], axis = 1), 1)
    z_min = np.expand_dims(np.min(all_landmarks[:,:,2], axis = 1), 1)

    all_landmarks[:,:,0] = (all_landmarks[:,:,0] - x_min) / (x_max - x_min)
    all_landmarks[:,:,1] = (all_landmarks[:,:,1] - y_min) / (y_max - y_min)
    all_landmarks[:,:,2] = (all_landmarks[:,:,2] - z_min) / (z_max - z_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks


def main(args):
    old_time = time.time()

    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    root_dir = config['dataset']['dataset_root_dir']

    test_pose_save_dir = os.path.join(root_dir, 'test_poses')
    test_video_dir = os.path.join(root_dir, 'video/test')
    label_dir = os.path.join(root_dir, 'annotation')
    if not os.path.exists(test_pose_save_dir):
        os.makedirs(test_pose_save_dir)

    label_name = 'test.csv'
    label_filename = os.path.join(label_dir, label_name)
    df = pd.read_csv(label_filename)

    for i in range(0, len(df)):
        filename = df.loc[i, 'name']

        video_path = os.path.join(test_video_dir, filename)
        test_pose_save_path = os.path.join(test_pose_save_dir, filename.replace('mp4', 'npy'))
        print('\nvideo input path:', video_path)
        print('test pose save path:', test_pose_save_path)

        video_cap = cv2.VideoCapture(video_path)
        # Get some video parameters.
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize tracker.
        pose_tracker = mp_pose.Pose()

        np_pose = []
        while True:
            # Get next frame of the video.
            success, frame = video_cap.read()
            if not success:
                break
            # Run pose tracker.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=frame)
            pose_landmarks = result.pose_landmarks

            if pose_landmarks is not None:
                pose_landmarks = np.array(
                    [[lmk.x * video_width, lmk.y * video_height, lmk.z * video_width]
                     for lmk in pose_landmarks.landmark],
                    dtype=np.float32)
                lanrmarks = np.expand_dims(pose_landmarks, axis=0)
                landmarks = normalize_landmarks(lanrmarks)
                landmarks = np.array(landmarks).astype(np.float32).reshape(-1)
            else:
                landmarks = np.zeros(99)
            np_pose.append(landmarks)

        np_pose = np.array(np_pose).astype(np.float32)
        np.save(test_pose_save_path, np_pose)

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)
