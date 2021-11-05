# Object-Detection-using-TF-lite-model-with-Tracking-Algorithm-SORT



A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences. See an example video here.

By Alex Bewley
Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in this paper. At the time of the initial publication, SORT was ranked the best open source multiple object tracker on the MOT benchmark.

Note: A significant proportion of SORT's accuracy is attributed to the detections. For your convenience, this repo also contains Faster RCNN detections for the MOT benchmark sequences in the benchmark format. To run the detector yourself please see the original Faster RCNN project or the python reimplementation of py-faster-rcnn by Ross Girshick.

Also see: A new and improved version of SORT with a Deep Association Metric implemented in tensorflow is available at https://github.com/nwojke/deep_sort .
License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@bewley.ai).
