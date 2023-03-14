# PoseRAC: Pose Saliency Transformer for Repetitive Action Counting
Here is the official implementation for paper "PoseRAC: Pose Saliency Transformer for Repetitive Action Counting"

<p align="center">
  <img src="images/JumpJack_demo.gif", width=380></a>
  <img src="images/Squat_demo.gif", width=380></a>
</p>

## Introduction
This code repo implements PoseRAC, the first pose-level network for Repetitive Action Counting. 

Repetitive action counting aims to count the number of repetitive actions in a video, while all current works on this task are video-level, which involves expensive feature extraction and sophisticated video-context interaction. On the other hand, human body pose is the most essential factor in an action, while it has not been well-explored in repetitive action counting task. Based on the motivations above, we propose the first pose-level method called **Pose** Saliency Transformer for **R**epetitive **A**ction **C**ounting (**PoseRAC**).

Meanwhile, the current datasets lack annotations to support pose-level methods, so we propose **Pose Saliency Annotation** to re-annotate the current best dataset *RepCount* to obtain the most representative poses for actions. We augment it with pose-level annotations, and create a new version: ***RepCount-pose***, which can be used by all future pose-level methods. We also make such enhancements on *UCFRep*, but this dataset lacks fine-grained annotations compared to *RepCount*, and has fewer actions for the healthcare and fitness fields, so we focus on the improvement of the *RepCount* dataset.

More details about the principles and techniques of our work can be found in the paper. Thanks!

Using Pose Saliency Annotation to train our PoseRAC, we achieve new state-of-the-art performance on *RepCount*, far outperforming all current methods, **with an OBO metric of 0.56 compared to 0.29 of previous state-of-the-art TransRAC!** Moreover, PoseRAC has a exaggerated running speed, which takes only 20 minutes to train on a single GPU, and it is even so lightweight to train in only one hour and a half on a CPU, which is unimaginable in previous video-level methods. Our method is also very fast during inference, which is almost 10x faster than the previous state-of-the-art method TransRAC on the average speed per frame.


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


## News

## RepCount-pose: A new version of RepCount dataset with pose-level annotations
We propose a novel **Pose Saliency Annotation** that addresses the lack of annotations for salient poses in current datasets. As figure below shows, take front raise action as an example, we pre-define two salient poses for each action and annotate the frame indices where these poses occur for all videos in the training set, creating new annotation files for our pose-level method to train on. We apply this approach to *RepCount*, and create a new annotated version called ***RepCount-pose***.

<p align="center">
  <img src="images/PSA.jpg", width=900></a>
</p>

#### Download Videos and Pose-level Annotations

## Code overview

## Usage

## Citation
If you find the project or the new version dataset is useful, please consider citing the paper.
