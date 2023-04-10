// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 2: Real time 2d Object Detection 


// RTOD.cpp
// All key options will be displayed at output menu


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include "obj_det.h"

int clr_spc = 0;

int main(int argc, char* argv[])
{
    cv::VideoCapture* vid;


    // open the video device
    std::string address = "http://10.110.17.157:4747/video";
    vid = new cv::VideoCapture(address);
    if (!vid->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }
   

    // get some properties of the image
    cv::Size refS((int)vid->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)vid->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);


    // Initialisation of some variables
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, dest;
    char QK = 1, key = -1, filt = 0;
    int bar_flag = 0;
    clr_spc = 0;
    std::string caption;

    //Generating random colors for segmentation visual
    std::vector<cv::Scalar> colors;
    rand_color_generator(colors);



    // Printing the Key Options
     std::cout << "\n ------Menu------\n"
        << "1. Threshold\n" << "2. Clean up"
        << "\n3. Segmentation" << "\n4. Bounding Boxes"
        << "\n5. Bounding Boxes with Moment axis"
        << "\n6. Frame to train from"
        << "\n7. Classify using database"
        << "\nW. Nearest neighbour classifier (by default) k=1"
        << "\nE. K = 2 Nearest neighbour classifier";
    printf("\n\n Enter option: ");
    // Main Video Loop for frame by frame
    while (QK) {
        *vid >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        } 
        cv::imshow("Video", frame);




        // see if there is a waiting keystroke
        key = cv::waitKey(10);
        if (key != -1) { // Only enters loop if a key is pressed and resets some flags
            filt = key;
            bar_flag = 0;
            cv::destroyAllWindows();
        }

        // switch case to check Key pressed and display output
        switch (key_check(frame, dest, filt, colors)) {

        case -1: break;    // if no key is pressed or if keys "g,h,b,x,y,m,l,c,n,e" are pressed
            // function calls happen in the key_check function


        case 2: cv::destroyAllWindows();  // if key 'q' is pressed ----- Stop runtime and close all windows
            return 0;

        case 3: filt = 5;
            break;
        }

    }   // end of switch case
    delete vid;
    return(0);
}


// End of program


