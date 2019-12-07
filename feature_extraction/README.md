# Data Extraction
We extracted data samples from [KTH](http://www.nada.kth.se/cvap/actions/) dataset.   

## Running

Ensure you have Video Dataset folder with following hierarchy:   
&nbsp;&nbsp;&nbsp;&nbsp;{Dataset folder}/{class label}/{video_file}<br/>
Example,  
&nbsp;&nbsp;&nbsp;&nbsp;VideoDataset/running/file1.avi
<br/>
<br/>

To run my code you need to type:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;python FeatureExtractor.py --video_data_dir <dataset_folder>

Here, argument <dataset_folder> is optional, and default value can be seen in the code.
<br/>
<br/>
In case you decided to stop program at some point, and wondering if it possible continue
from that moment - the answer is, yes!<br/>
Just stop program and run it again, it will continue extracting points approximately from
last interruption.

## Output
Basically, Program outputs a following folder hierarchy:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;{Dataset folder}/{class label}/data.txt<br/>
<br/>
This folder then should be placed in the {TrainActivityRecognition} folder. 

## Data Sample
Body Part mapping is located in <pose_estimation/common.py><br/>
Needed body parts for extraction can be edited at the beginning of <Detection.py>.
<br/>
We chose 14 key point data for extraction, thinking that with this data, it is 
possible to fully recognize defined actions.<br/>
<b>Thus, Data consists of x,y coords of 14 body parts in {n} frames. Where {n} is 10.</b><br/><br/>
The reasoning behind value for {n}, is that we extract key-points data only each 2 or 3 
frames. Therefore,
10 frames will map to the 20-30 frames in real video. If our program can process only 10-15 fps then it means 
that data will cover 1-3 seconds of video in real-time. We suggest, that this amount of data is enough to 
recognize simple actions like in our dataset.  
Evidently,these numbers are not something we tested very much, so, feel free to experiment with these numbers.<br/>
<br/>
<br/>Consequently,Single Data Sample Length:
<p align="center">
<br/>  
290 = 2*14*10 (x,y coords of 14 body parts in 10 frames) + 10(indexes of each frame)
</p>
