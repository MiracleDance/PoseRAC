# PoseRAC: Pose Saliency Transformer for Repetitive Action Counting

<p align="center">
  <a href="https://arxiv.org/abs/2303.08450"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 
</p>

<p align="center">
  <a href="https://paperswithcode.com/sota/repetitive-action-counting-on-repcount?p=poserac-pose-saliency-transformer-for"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/poserac-pose-saliency-transformer-for/repetitive-action-counting-on-repcount" ></a> </h1> 
</p>

Here is the official implementation for paper <a href="https://arxiv.org/abs/2303.08450">PoseRAC: Pose Saliency Transformer for Repetitive Action Counting</a>.

<p align="center">
  <a> <img src="images/JumpJack_demo.gif" width='42%'></a>   <a> <img src="images/Squat_demo.gif" width='42%'></a>
</p>

Our PoseRAC achieves a substantial improvement over the previous state-of-the-art in performance and speed, even having significantly fewer model parameters, which represented by the sizes of bubbles.
<p align="center">
  <img src="images/bubble.jpg" width='50%'>
</p>

## Introduction
This code repo implements PoseRAC, the first pose-level network for Repetitive Action Counting. 

Repetitive action counting aims to count the number of repetitive actions in a video, while all current works on this task are video-level, which involves expensive feature extraction and sophisticated video-context interaction. On the other hand, human body pose is the most essential factor in an action, while it has not been well-explored in repetitive action counting task. Based on the motivations above, we propose the first pose-level method called **Pose** Saliency Transformer for **R**epetitive **A**ction **C**ounting (**PoseRAC**).

Meanwhile, the current datasets lack annotations to support pose-level methods, so we propose **Pose Saliency Annotation** to re-annotate the current best dataset *RepCount* to obtain the most representative poses for actions. We augment it with pose-level annotations, and create a new version: ***RepCount-pose***, which can be used by all future pose-level methods. We also make such enhancements on *UCFRep*, but this dataset lacks fine-grained annotations compared to *RepCount*, and has fewer actions for the healthcare and fitness fields, so we focus on the improvement of the *RepCount* dataset.

***More details about the principles and techniques of our work can be found in the paper. Thanks!***

Using Pose Saliency Annotation to train our PoseRAC, we achieve new state-of-the-art performance on *RepCount*, far outperforming all current methods, **with an OBO metric of 0.56 compared to 0.29 of previous state-of-the-art TransRAC!** Moreover, PoseRAC has an exaggerated running speed, which takes only 20 minutes to train on a single GPU, and it is even so lightweight to train in only one hour and a half on a CPU, which is unimaginable in previous video-level methods. Our method is also very fast during inference, which is almost 10x faster than the previous state-of-the-art method TransRAC on the average speed per frame.


|        Methods       |  MAE $\downarrow$  |  OBO $\uparrow$ | Time(ms) |
|:--------------------:|:-----:|:-----:|:--------:|
|        RepNet        | 0.995 | 0.013 |    100   |
|          X3D         | 0.911 | 0.106 |    220   |
|     Zhang et al.     | 0.879 | 0.155 |    225   |
|         TANet        | 0.662 | 0.099 |    187   |
| VideoSwinTransformer | 0.576 | 0.132 |    149   |
|     Huang et al.     | 0.527 | 0.159 |    156   |
|       TransRAC       | 0.443 | 0.291 |    200   |
|     **PoseRAC(Ours)**    | **0.236** | **0.560** |    **20**    |

## RepCount-pose: A new version of RepCount dataset with pose-level annotations
We propose a novel **Pose Saliency Annotation** that addresses the lack of annotations for salient poses in current datasets. As figure below shows, take front raise action as an example, we pre-define two salient poses for each action and annotate the frame indices where these poses occur for all videos in the training set, creating new annotation files for our pose-level method to train on. We apply this approach to *RepCount*, and create a new annotated version called ***RepCount-pose***.

<p align="center">
  <img src="images/PSA.jpg" width='80%'>
</p>

#### Download Videos and Pose-level Annotations
[this Google Drive link](https://drive.google.com/file/d/1k9LLzOsJVh6ACXSX8iKbGNxTY9-L6X_x/view?usp=sharing)


## Code overview
* After preparing the dataset above, the folder structure should look like:
```
This folder
│   README.md
│   best_weights_PoseRAC.pth
|   train.py
|   eval.py
│   ...

└───RepCount_pose/
│    └───annotation/
│    │	 └───pose_train.csv
│    │	 └───test.csv  
│    │   └───valid.csv 
│    │   └───video_train.csv
│    └───original_data/
│    └───test_poses/
│    └───video/
│    │	 └───test/
│    │	 └───train/
│    │   └───valid/
```
It is worth mentioning that in the *./RepCount_pose/annotation/* directory, there are two files for training, where *pose_train.csv* is the annotation applied to our pose-level method, while *video_train.csv* is applied to the common video-level method. Other than that, there is no difference between our *RepCount-pose* and *RepCount*, including videos, test annotations, etc.

## Usage
### Install
Please refer to INSTALL.md for installation, or you can use:
```sh
pip install -r requirement.txt
```

### Evaluation
- [**Optional**] Obtain the pose for each frame of each test video. 
- As all poses of the test videos have been extracted already by us (see the *./RepCount_pose/test_poses/*), you can ignore this step! Or you can also try this step, the purpose is only to generate all the data in *./RepCount_pose/test_poses/*.

```sh
python pre_test.py --config ./RepCount_pose_config.yaml
```

- Evaluate our PoseRAC with pretrained checkpoint:
```sh
python eval.py --config ./RepCount_pose_config.yaml --ckpt ./best_weights_PoseRAC.pth
```
- Then, you can get the evaluation results:
```
MAE:0.2356079854110582, OBO:0.5592105263157895
```

### Training
- Preprocessing before training. According to the pose-level annotation, the key frames of each video are determined, and we extract the poses of these frames and get their corresponding classes.

```sh
python pre_train.py --config ./RepCount_pose_config.yaml
```

- Train the model:
```sh
python train.py --config ./RepCount_pose_config.yaml
```

### Inference and Visualization
```sh
python inference_and_visualization.py --config ./RepCount_pose_config.yaml --ckpt ./best_weights_PoseRAC.pth
```

You can also train from scratch to get a set of model weights for evaluation and inference.

## Contact
Ziyu Yao (yaozy@stu.pku.edu.cn)

If you have any questions or suggestions, don't hesitate to contact us!

## Citation
If you are using our code or new version dataset, please consider citing our paper.
```
@article{yao2023poserac,
  title={PoseRAC: Pose Saliency Transformer for Repetitive Action Counting},
  author={Yao, Ziyu and Cheng, Xuxin and Zou, Yuexian},
  journal={arXiv preprint arXiv:2303.08450},
  year={2023}
}
```
