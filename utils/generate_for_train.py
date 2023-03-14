import os
import cv2
import csv
from mediapipe.python.solutions import pose as mp_pose
import numpy as np


# For the selected key frames, we use the pose estimation network to extract the poses.
# For each pose, we use 33 key points to represent it, and each key point has 3 dimensions.
def _generate_for_train(root_dir):
    data_folder = os.path.join(root_dir, 'extracted')
    out_csv_dir = os.path.join(root_dir, 'annotation_pose')

    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)

    for train_type in os.listdir(data_folder):
        if '.DS_Store' in train_type:
            continue
        out_csv_path = os.path.join(out_csv_dir, train_type) + '.csv'
        with open(out_csv_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            sub_train_folder = os.path.join(data_folder, train_type)
            for action_type in os.listdir(sub_train_folder):
                sub_sub_folder = os.path.join(sub_train_folder, action_type)
                print(action_type)
                if '.DS_Store' in action_type:
                    continue
                for salient1_2 in os.listdir(sub_sub_folder):
                    sub_sub_sub_folder = os.path.join(sub_sub_folder, salient1_2)
                    if '.DS_Store' in salient1_2:
                        continue
                    for video_name in os.listdir(sub_sub_sub_folder):
                        video_dir = os.path.join(sub_sub_sub_folder, video_name)
                        if '.DS_Store' in video_dir:
                            continue
                        for single_path in os.listdir(video_dir):
                            if '.DS_Store' in single_path:
                                continue
                            if '.jpg' not in single_path:
                                continue
                            image_path = os.path.join(video_dir, single_path)
                            base_path = os.path.join(train_type, action_type, salient1_2,  video_name, single_path)
                            input_frame = cv2.imread(image_path)
                            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                            # Initialize fresh pose tracker and run it.
                            with mp_pose.Pose() as pose_tracker:
                                result = pose_tracker.process(image=input_frame)
                                pose_landmarks = result.pose_landmarks
                            output_frame = input_frame.copy()
                            # Save landmarks if pose was detected.
                            if pose_landmarks is not None:
                                # Get landmarks.
                                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                                pose_landmarks = np.array(
                                    [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                     for lmk in pose_landmarks.landmark],
                                    dtype=np.float32)
                                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                                csv_out_writer.writerow([base_path, action_type] + pose_landmarks.flatten().astype(str).tolist())
