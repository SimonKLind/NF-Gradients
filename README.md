# NF-Gradients
This repository contains code and images accompanying the publication "<I>Making the Flow Glow - Robot Perception under Severe Lighting Conditions using Normalizing Flow Gradients</I>" (S. Kristoffersson Lind, R. Triebel, V. Krüger, presented at IROS2024).

# Runnable files
The following files should be runnable with little effort (with dependencies: PyTorch, TorchVision, OpenCV, NumPy), as long as you have an appropriate dataset of images:
 - frcnn_features.py: This file will run a ResNet-based Faster-RCNN from TorchVision, and extract features that can subsequently be used to train a Normalizing Flow.
 - train_NF.py: This file will train a Normalizing Flow on features extracted by frcnn_features.py.
 - eval_frcnn_NF.py: This file will run visualizations of the Normalizing Flow gradients, like the ones in our paper.

# Non-Runnable files
The following files are not assumed to be runnable, as they are <B>heavily</B> specialized for our robot setup (we include them here only for completeness):
 - realsense_ros_custom.py: a custom ROS-wrapper that allows us to get images from our Realsense camera, and set parameters.
 - realsense_skills.py: our main experiment files, implemented as skills within the Skiros2 framework.

Points of interest for anyone wishing to reproduce our work include:
 - realsense_ros_custom.py, lines 46-54: the exact parameters we use in our Realsense D435 camera.
 - realsense_skills.py, lines 347-380: our evolutionary optimization routine.
 - realsense_skills.py, lines 262-300: the function we optimize over.
