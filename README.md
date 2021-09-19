# Unity ML pose detection sandbox with Mike
Made in Unity v.2020.03.18f1</br>

![Yoga.png](Assets/Screenshots/Yoga.png)

## Outline
UML_skel3D is a ML-based multi-pose estimation made in Unity using barracuda, relying on precomputed models (onnx). Ironically, the 3D portion doesn't work yet, but I felt ambitious when the title was chosen. </br>

The project consists of posenet implementations using currently has 4 working models:
- MobileNet
- ResNet50
- DeepMobileNet (more layers added to MobileNet) -> This model was computed from a pytorch-based checkpoint from https://awesomeopensource.com/project/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
- OpenPose (works only in single-pose estimation)

There is a post-processing involved to temporarily smooth the predictions using a kalman filter applied on the keypoints to reduce the jitters. This can be tuned-up on the fly.

Most models performance could not be evaluated on GPU (only CPU), but should support it. On a good note, it works surprisingly fine on a macbook pro (SSD). </br>

The code to generate the DeepMobileNet is in "Assets/Models/pytorch_to_onnx".


## A few results

. Kalman effect demonstrated by deactivating or varying the parameters (using MobileNet):
https://user-images.githubusercontent.com/35206039/133916271-c8a0db4d-2e2f-49c7-aa33-c4707af078e7.mov

. Single-pose (using DeepMobileNet with the Kalman filter activated):
https://user-images.githubusercontent.com/35206039/133916287-8055c53a-6bea-4d1d-9d3e-a7fff539914e.mov

. Single-pose (using DeepMobileNet with the Kalman filter activated):
https://user-images.githubusercontent.com/35206039/133916287-8055c53a-6bea-4d1d-9d3e-a7fff539914e.mov

. Multi-pose (DeepMobileNet with the Kalman filter activated):
https://user-images.githubusercontent.com/35206039/133916526-15ba368d-5775-4b6d-ad89-385bdb3cf348.mp4
https://user-images.githubusercontent.com/35206039/133916524-7c02ced7-35ca-4d98-8aaa-5796195959ea.mp4


------------

## Install
### Download and fill the missing files

1. Download/clone the repo locally, install Unity and Barracuda.</br>

2. Download the models from here, and put them in the 'Assets/Models' folder.</br>
https://drive.google.com/file/d/11Ry-VCO-epLli8cgFH_GPjVz6-HqU4xb/view?usp=sharing
 
### Settings in Unity Inspector

1. Drag the "ImRickJames" from the ""Assets/Scene" folder to the Hierarchy view.</br>

2. Choose a video from "Assets/Videos", and drag it to the "VideoScreen" game object ('Video Player/Video Clip') </br>

3. Drag the "PoseEstimation.cs" from "Assets/Script" and drag it to the "PoseEstimator" game object (where it says 'missing script'). </br>

This should open the other fields to fill:

<img src="Assets/Screenshots/PoseEstimator.png" width="500">

4. Fill the missing information to be similar to the screenshot

. Video Screen: Choose Video Screen (duh)
. PoseNet shader: Choose PreprocessShader
. Assets: The ONNX files from the "Assets/Models" folder. 

## How to use the estimator

# Parameters to set in 'PoseEstimator' game object:

. Model Type: Choose one of the four models proposed (OpenPose does not support multi-pose)
. Use GPU: Flag to select if the computation will be made on the GPU or CPU (mainly tested on a macos SSD CPU)
. ImageDims: How the original images will be resized before feeding it to the model. High means more processing power, but better results. ZThe aspect ratio will be adjusted based on the model automaticallly.
. Estimation Type: Single pose or Multi-Pose. Multi-Pose with a single target works, but is less reliable (jittery).
  

## License
### Non-commercial use</br>
・Please use it freely for research. </br>

</br></br>
  
