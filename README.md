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

## News

## RepCount-pose Dataset
We propose a novel **Pose Saliency Annotation** that addresses the lack of annotations for salient poses in current datasets. As figure below shows, we pre-define two salient poses for each action and annotate the frame indices where these poses occur for all videos in the training set, creating new annotation files for our pose-level method to train on. We apply this approach to *RepCount*, and create a new annotated version called ***RepCount-pose***.

## Code Overview

## Usage

## Citation
If you find the project or the new version dataset is useful, please consider citing the paper.
