<!-- # ActionRecognition
### Action Recognition based on pose estimation.  
**Graduation Project for Bachelor Degree**</br> 
By Kenessary Koishybay, Nauryzbek Razakhbergenov.</br>
Mentor: Anara Sandygulova. **Nazarbayev University**


## Introduction
Pose estimation algortihm is based on [tensorflow implementation](https://github.com/ildoonet/tf-pose-estimation) of
[Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)   

<p align="left">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/asd1.gif", width="400">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/asd2.gif", width="400">
</p>
<p align="left">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/demo.gif", width="400">
</p>


Requirements:
- Python 2.7  
- OpenCV3  
- sklearn  
- scipy  
- imutils  
- xgboost    
	
	
## Running
To run my code you need to type:  
&nbsp;&nbsp;&nbsp;&nbsp;python -B Main.py &lt;input_video&gt; &lt;output_video&gt;  
Here, arguments <input_video> and <output_video> are optional, 
and default values can be seen in the code.
  
## How it works  


### Pose Estimation
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/pose.png">
</p>

Pose estimation is the process of locating body key points.<br/>
Pose estimation problem is usually solved by training Deep Learning architectures with annotated datasets such as 
<a href="http://human-pose.mpi-inf.mpg.de">MPII</a> or <a href="http://cocodataset.org/">COCO</a> 
<br/>
We didn't have computational power to train on these datasets. Thus, we tried pre-trained model mentioned at the beginning.
<br/>
Architecture:
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/arch.jpg">
</p>
However, after looking that even prediction on that architecture takes too much time, we use Mobile Net
in the final version.
We use pose estimation for Detection and collecting coordinates (x,y) of body key-points. 

### Tracking
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/track.jpg">
</p>
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/track_alg.jpg">
</p>


Note, that we decided to remove code concerning **EWMA** in the final version.

### Activity Recognition 

<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/recog.png">
</p>

<p align="center">
Single Data Sample Length:<br/>  
290 = 2*14*10 (x,y coords of 14 body parts in 10 frames) + 10(indexes of each frame)
</p>

For every (N = 10)th frame:
1. Open pose features calculated for every tracked humans
2. This features is then added to the previous features of the tracks
3. If the length of resulting feature vectors of specific tracks is large enough, feature vectors will be converted to data samples
4. These data samples is then goes as input to the machine learning algorithm (XGBoost)
5. XGBoost classifies activity of each data sample as code.
6. Code is then decoded into Activity Labels




## Training
If you wan't to train our activity recognition algorithm to increase accuracy, first you should **extract suitable data** from videos dataset.

### Data Extraction

We extracted data samples from [KTH](http://www.nada.kth.se/cvap/actions/) dataset.   
Code is in ExtractData folder.


### Model Selection and Training


## TODO


 -->