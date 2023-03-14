# PoseRAC: Pose Saliency Transformer for Repetitive Action Counting
Here is the official implementation for paper "PoseRAC: Pose Saliency Transformer for Repetitive Action Counting"

<p align="center">
<img src="https://github.com/MiracleDance/PoseRAC/blob/main/images/MyVideo_1.gif?raw=true", width=360></a>  
<img src="https://github.com/MiracleDance/PoseRAC/blob/main/images/MyVideo_2.gif?raw=true", width=360></a>
</p>

## Introduction
This code repo implements PoseRAC, the first pose-level network for Repetitive Action Counting. 

Repetitive action counting aims to count the number of repetitive actions in a video, while all current works on this task are video-level, which involves expensive feature extraction and sophisticated video-context interaction. On the other hand, human body pose is the most essential factor in an action, while it has not been well-explored in repetitive action counting task. Based on the motivations above, we propose the first pose-level method called **Pose** Saliency Transformer for **R**epetitive **A**ction **C**ounting (**PoseRAC**).

Meanwhile, the current datasets lack annotations to support pose-level methods, so we propose Pose Saliency Annotation to re-annotate the current best dataset *RepCount* to obtain the most representative poses for actions. We augment it with pose-level annotations, and create a new version: ***RepCount-pose***, which can be used by all future pose-level methods. We also make such enhancements on *UCFRep*, but this dataset lacks fine-grained annotations compared to *RepCount*, and has fewer actions for the healthcare and fitness fields, so we focus on the improvement of the *RepCount* dataset.

More details about the principles and techniques of our work can be found in the paper. Thanks!


