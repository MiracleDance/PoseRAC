import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import io
import argparse
import yaml
from model import PoseRAC, Action_trigger

torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

    z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
    z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)

    all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
    all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)
    all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


class PoseClassificationVisualizer(object):
    """Keeps track of claassifcations for every frame and renders them."""
    def __init__(self,
                 class_name,
                 plot_location_x=0.05,
                 plot_location_y=0.05,
                 plot_max_width=0.4,
                 plot_max_height=0.4,
                 plot_figsize=(9, 4),
                 plot_x_max=None,
                 plot_y_max=None,
                 counter_location_x=0.85,
                 counter_location_y=0.05,
                 counter_font_color='red',
                 counter_font_size=0.15):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size
        self._counter_font = None
        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with classification plot and counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the plot.
        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS)
        output_img.paste(img,
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y)))

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            self._counter_font = ImageFont.truetype('Roboto-Regular.ttf', size=font_size)
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),
                             font=self._counter_font,
                             fill=self._counter_font_color)

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)
        for classification_history in [self._pose_classification_history,
                                       self._pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                else:
                    y.append(classification)
            plt.plot(y, linewidth=7)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.title('Classification history for `{}`'.format(self._class_name))

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


def main(args):
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path']
    root_dir = config['dataset']['dataset_root_dir']
    output_video_dir = os.path.join(root_dir, 'video_visual_output', 'test')
    input_video_dir = os.path.join(root_dir, 'video', 'test')
    poses_save_dir = os.path.join(root_dir, 'test_poses')

    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)

    test_csv_name = os.path.join(root_dir, 'annotation', 'test.csv')
    test_df = pd.read_csv(test_csv_name)

    label_pd = pd.read_csv(csv_label_path)
    index2action = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index2action[label] = action
    num_classes = len(index2action)
    print(index2action)

    model = PoseRAC(None, None, None, None, dim=config['PoseRAC']['dim'], heads=config['PoseRAC']['heads'],
                    enc_layer=config['PoseRAC']['enc_layer'], learning_rate=config['PoseRAC']['learning_rate'],
                    seed=config['PoseRAC']['seed'], num_classes=num_classes, alpha=config['PoseRAC']['alpha'])
    # model.load_from_checkpoint(weight_path)
    weight_path = 'best_weights_PoseRAC.pth'
    new_weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(new_weights)
    model.eval()

    enter_threshold = config['Action_trigger']['enter_threshold']
    exit_threshold = config['Action_trigger']['exit_threshold']
    momentum = config['Action_trigger']['momentum']

    for i in range(0, len(test_df)):
        video_name = test_df.loc[i, 'name']
        gt_count = test_df.loc[i, 'count']

        poses_save_path = os.path.join(poses_save_dir, video_name.replace('mp4', 'npy'))

        all_landmarks = np.load(poses_save_path).reshape(-1, 99)
        all_landmarks_tensor = torch.from_numpy(all_landmarks).float()
        all_output = torch.sigmoid(model(all_landmarks_tensor))

        best_mae = float('inf')
        real_action = 'none'
        real_index = -1
        for index in index2action:
            action_type = index2action[index]
            # Initialize action trigger.
            repetition_salient_1 = Action_trigger(
                action_name=action_type,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold)
            repetition_salient_2 = Action_trigger(
                action_name=action_type,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold)

            classify_prob = 0.5
            pose_count = 0
            curr_pose = 'holder'
            init_pose = 'pose_holder'
            for output in all_output:
                output_numpy = output[index].detach().cpu().numpy()
                classify_prob = output_numpy * (1. - momentum) + momentum * classify_prob
                # Count repetitions.
                salient1_triggered = repetition_salient_1(classify_prob)
                reverse_classify_prob = 1 - classify_prob
                salient2_triggered = repetition_salient_2(reverse_classify_prob)

                if init_pose == 'pose_holder':
                    if salient1_triggered:
                        init_pose = 'salient1'
                    elif salient2_triggered:
                        init_pose = 'salient2'

                if init_pose == 'salient1':
                    if curr_pose == 'salient1' and salient2_triggered:
                        pose_count += 1
                else:
                    if curr_pose == 'salient2' and salient1_triggered:
                        pose_count += 1

                if salient1_triggered:
                    curr_pose = 'salient1'
                elif salient2_triggered:
                    curr_pose = 'salient2'

            mae = abs(gt_count - pose_count) / (gt_count + 1e-9)
            if mae < best_mae:
                best_mae = mae
                real_action = action_type
                real_index = index

        action_type = real_action

        # Initialize action trigger.
        repetition_salient_1 = Action_trigger(
            action_name=action_type,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold)
        repetition_salient_2 = Action_trigger(
            action_name=action_type,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold)
        classify_prob = 0.5
        pose_count = 0
        curr_pose = 'holder'
        init_pose = 'pose_holder'
        video_path = os.path.join(input_video_dir, video_name)
        output_video_path = os.path.join(output_video_dir, video_name)
        print('video input path', video_path)
        print('video output path', output_video_path)

        video_cap = cv2.VideoCapture(video_path)
        # Get some video parameters to generate output video with classificaiton.
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Initilize tracker, classifier and counter.
        # Do that before every video as all of them have state.
        # Initialize tracker.
        pose_tracker = mp_pose.Pose()
        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                                    (video_width, video_height))

        pose_classification_visualizer = PoseClassificationVisualizer(
            class_name=action_type,
            plot_x_max=video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10)
        frame_idx = 0
        frame_count = 0
        for output in all_output:
            success, input_frame = video_cap.read()
            if not success:
                break
            frame_count += 1
            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
            output_numpy = output[real_index].detach().cpu().numpy()
            classify_prob = output_numpy * (1. - momentum) + momentum * classify_prob
            # Count repetitions.
            salient1_triggered = repetition_salient_1(classify_prob)
            reverse_classify_prob = 1 - classify_prob
            salient2_triggered = repetition_salient_2(reverse_classify_prob)

            if init_pose == 'pose_holder':
                if salient1_triggered:
                    init_pose = 'salient1'
                elif salient2_triggered:
                    init_pose = 'salient2'

            if init_pose == 'salient1':
                if curr_pose == 'salient1' and salient2_triggered:
                    pose_count += 1
            else:
                if curr_pose == 'salient2' and salient1_triggered:
                    pose_count += 1

            if salient1_triggered:
                curr_pose = 'salient1'
            elif salient2_triggered:
                curr_pose = 'salient2'

            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=classify_prob,
                pose_classification_filtered=classify_prob,
                repetitions_count=pose_count)

            output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)

            save_picture = output_frame.copy()
            frame_idx += 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (int(video_width * 0.1), int(video_height * 0.9))
            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 3

            # Using cv2.putText() method
            show_text = 'action: {}'.format(action_type)
            save_picture = cv2.putText(save_picture, show_text, org, font,
                                       fontScale, color, thickness, cv2.LINE_AA)

            out_video.write(save_picture)

        # Release MediaPipe resources.
        pose_tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    args = parser.parse_args()
    main(args)
