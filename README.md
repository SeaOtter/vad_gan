# vad_gan
# Robust Anomaly Detection in Videos using Multilevel Representations

This project proposed a anomaly detection framework using multilevel information to identify anomaly objects in surveillance videos. Our research results in re-annotating UCSD Ped 1 dataset (http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm original dataset) which is one of the most widely-used datasets in Video Anomaly Detection. You can find the new label set in relabeled_ped1.zip.

## New label set for UCSD Ped 1

### Folder structure: 
+ relabeled_frames_anno
	|------Test001_anno	
	|		|---053.bmp
	|		|---053_anno.bmp
	|		|---054.bmp
	|		|---054_anno.bmp
	|		|	...
	|		|---069.bmp
	|		|---069_anno.bmp
	|
	|------Test002_anno
	|		|...
	|
	|------Test003_anno
	|		|...
	|	...
	|------Test036_anno
			|...
+ relabeled_frames_all
	|------Test001_gt
	|	|------001.bmp
	|	|------002.bmp
	|	|	...
	|	|------200.bmp
	|
	|------Test002_gt
	|	...
	|------Test0036_gt

### Explanation: 
*relabeled_frames_anno: only consists of frames with label modifications. Each subfolder is a video and only videos with label modifications are shown in this folder. 
Each frame has two *.bmp files: 
	a) *_anno.bmp is the ground-truth of additional anomaly objects
	b) *.bmp is the new ground-truth that is the OR image of old UCSD Ped 1 ground-truth and additional ground-truth (*_anno.bmp)
*relabeled_frames_all: is the relabeled ground-truth for all frame in UCSD Ped 1 that are the new ground-






