import pandas as pd
import numpy as np
import os
import torch.onnx
from model import PoseRAC, Action_trigger
import argparse
import yaml
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path']
    root_dir = config['dataset']['dataset_root_dir']

    test_pose_save_dir = os.path.join(root_dir, 'test_poses')
    test_video_dir = os.path.join(root_dir, 'video/test')
    label_dir = os.path.join(root_dir, 'annotation')

    label_pd = pd.read_csv(csv_label_path)
    index2action = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index2action[label] = action
    num_classes = len(index2action)

    label_filename = os.path.join(label_dir, 'test.csv')
    df = pd.read_csv(label_filename)

    model = PoseRAC(None, None, None, None, dim=config['PoseRAC']['dim'], heads=config['PoseRAC']['heads'],
                    enc_layer=config['PoseRAC']['enc_layer'], learning_rate=config['PoseRAC']['learning_rate'],
                    seed=config['PoseRAC']['seed'], num_classes=num_classes, alpha=config['PoseRAC']['alpha'])
    assert args.ckpt is not None, 'checkpoint file does not exist'
    weight_path = args.ckpt
    new_weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(new_weights)
    model.eval()
    model.cuda()

    testMAE = []
    testOBO = []
    enter_threshold = config['Action_trigger']['enter_threshold']
    exit_threshold = config['Action_trigger']['exit_threshold']
    momentum = config['Action_trigger']['momentum']

    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        gt_count = df.loc[i, 'count']

        video_path = os.path.join(test_video_dir, filename)
        test_pose_save_path = os.path.join(test_pose_save_dir, filename.replace('mp4', 'npy'))
        print('\nvideo input path', video_path)

        poses = np.load(test_pose_save_path).reshape(-1, config['PoseRAC']['all_key_points'])
        poses_tensor = torch.from_numpy(poses).float()
        all_output = torch.sigmoid(model(poses_tensor.cuda()))
        # all_output = model(poses_tensor.cuda())

        # action_counts = [0] * num_classes
        # all_classes = torch.argmax(all_output, dim=1).view(-1, 1)
        # all_class_int = all_classes.cpu().numpy().flatten()
        # all_prob_class = torch.gather(all_output, dim=1, index=all_classes).detach().cpu().numpy()
        # larger_than_thresh = (all_prob_class > enter_threshold).flatten()
        # all_include_class_int = all_class_int[larger_than_thresh]
        # for class_idx in all_include_class_int:
        #     action_counts[class_idx] += 1
        # action_index = np.argmax(action_counts)
        # most_action = index_label_dict[action_index]
        # action_type = most_action

        best_mae = float('inf')
        best_obo = -float('inf')
        for index in index2action:
            action_type = index2action[index]
            # Initialize counter.
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
            if abs(gt_count - pose_count) <= 1:
                obo = 1
            else:
                obo = 0
            if mae < best_mae:
                best_mae = mae
                best_obo = obo

        testMAE.append(best_mae)
        testOBO.append(best_obo)

    print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    args = parser.parse_args()
    main(args)
