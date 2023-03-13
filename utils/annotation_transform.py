import numpy as np
import os
import cv2
import pandas as pd

def _annotation_transform(root_dir):
    train_type = 'train'
    annotation_name = 'pose_train_all.csv'
    video_dir = os.path.join(root_dir, 'video', train_type)
    label_filename = os.path.join(root_dir, 'annotation', annotation_name)
    train_save_dir = os.path.join(root_dir, 'extracted')
    save_dir = os.path.join(train_save_dir, train_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    df = pd.read_csv(label_filename)

    file2label = {}
    num_idx = 3
    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        action_type = df.loc[i, 'type']
        label_tmp = df.values[i][num_idx:].astype(np.float64)
        label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
        start_tmp = label_tmp[::2]
        end_tmp = label_tmp[1::2]
        assert len(label_tmp) % 2 == 0
        file2label[filename] = [start_tmp, end_tmp, action_type]

    for video_name in file2label:
        video_path = os.path.join(video_dir, video_name)
        print('video_path:', video_path)
        cap = cv2.VideoCapture(video_path)
        frames = []
        if cap.isOpened():
            while True:
                success, frame = cap.read()
                if success is False:
                    break
                frames.append(frame)
        cap.release()
        start_label, end_label, action_type = file2label[video_name]
        count = 0
        for frame_index in start_label:
            if frame_index >= len(frames):
                continue
            frame_ = frames[frame_index]
            sub_start_save_dir = os.path.join(save_dir, action_type, 'start', video_name)
            if not os.path.isdir(sub_start_save_dir):
                os.makedirs(sub_start_save_dir)
            save_path = os.path.join(sub_start_save_dir, str(count) + '.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1

        for frame_index in end_label:
            if frame_index >= len(frames):
                continue
            frame_ = frames[frame_index]
            sub_start_save_dir = os.path.join(save_dir, action_type, 'end', video_name)
            if not os.path.isdir(sub_start_save_dir):
                os.makedirs(sub_start_save_dir)
            save_path = os.path.join(sub_start_save_dir, str(count) + '.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1
