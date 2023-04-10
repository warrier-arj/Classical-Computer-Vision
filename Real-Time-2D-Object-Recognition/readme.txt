Real-time 2-D Object Recognition Project by Arjun Rajeev Warrier

Introduction:
This project aims to develop a program that identifies objects placed on a white background in real-time. The program utilizes a video stream to detect objects and uses thresholding, cleaning up binary images, segmenting images into regions, and computing features for each major region. The program also collects training data, classifies new images using the nearest neighbor function and k-nearest neighbor classifier, and evaluates performance using confusion tables. Currently, multiple objects are detected but not classified.

Usage:
To run the program, a video stream should be available. The user can select the frame to consider as a test or train case by pressing a button. The program will display the recognized object name in a separate window in the middle of the object bounding box.The port for camera will have to be changed in the code prior to running. 
Videos have been attached along with report and code files.

Menu will be displayed on running.

Requirements:

C++ programming language
OpenCV library
Notes:

The program demonstrates the effect of preprocessing and cleanup of thresholded images before usage in classification.
The project shows the pipeline of object detection and how each step in the pipeline weighs in heavily.
The project provides insight into how segmentation algorithms may work.

Author:
Arjun Rajeev Warrier
 
