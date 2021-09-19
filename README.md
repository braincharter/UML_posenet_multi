# UML_skel3D


## Outline
UML_skel3D is a ML-based multi-pose estimation made in Unity using barracuda, relying on precomputed models (onnx). Ironically, the 3D portion doesn't work yet, but I felt ambitious when the title was chosen. </br>

![Yoga.png](Assets/ScreenShots/Yoga.png)

The project consists of posenet implementations using currently has 4 working* models:
- MobileNet
- ResNet50
- LightMobileNet (less layers)
- OpenPose (works only in single-pose estimation)

Most models performance could not be evaluated on GPU (only CPU), but should support it. On a good note, it works surprisingly fine on a macbook pro (SSD). </br>

This was made in Unity v.2020.03.18f1</br>


## Install
### Download and fill the missing files

1. Download/clone the repo locally, install Unity and Barracuda.</br>

2. Download the models from here, and put them in the 'Assets/Models' folder.</br>
 
### Settings in Unity Inspector

1. Drag the "ImRickJames" from the "Scene" folder to the Hierarchy view.</br>

2. Choose a video from 'Assets/Videos', and drag it the the 'VideoScreen' game object ('Video Player/Video Clip') </br>
   
3. Set models</br>
   Drag the corresponding NNModel you put before in Assets/Scripts/Model/ 
   and drop it to respective "NN Model" in Gameobject named as "PoseEstimator" in Hierarchy view.
   

## Tutorial<br>
・ Choose Video</br>
   You can choose the target video.</br>
   Put the video you choose in Assets/Video/, and then drag the file and drop it to Video Clip of "Video Player".<br>
   ![unity_chooseVideo.PNG](Assets/StreamingAssets/ScreenShots/unity_chooseVideo.PNG)
   
・Choose Avatar</br>
    There are two types of avatar in this Scene.</br>
    You can change the avatar easily in inspector view.</br>
    Firstly activate Gameobject named as "Tait" and deactivate "unitychan".</br>
    Secondly drag the Gameobject and drop it to "V Nect Model" of "BarracudaRunner".</br>
    ![unity_set_anoter_avater_to_obj.PNG](Assets/StreamingAssets/ScreenShots/unity_set_anoter_avater_to_obj.PNG)</br>
    
    *To determin direction of the face of avatar, a gameobject which works as nose has been added in those of avatars.
     So if you would like to adopt your original avatar, please add the nose referencing the code.
     
・Use Web Camera
   By checking "Use Web Cam", you can change the input images.</br>
   ![unity_use_web_cam.PNG](Assets/StreamingAssets/ScreenShots/unity_use_web_cam.PNG)</br>
   
・Skip On Drop</br>
   If "Skip On Drop" in Video Player checked, VideoPlayer is allowed to skip frames to catch up with current time.<br>


## License
### Non-commercial use</br>
・Please use it freely for research. </br>

</br></br>
  
