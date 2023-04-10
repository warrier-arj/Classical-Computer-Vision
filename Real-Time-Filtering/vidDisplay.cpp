// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 1:  Real-time filtering 


//vidDisplay.cpp to carry out real time video filtering
// All key options will be displayed at output menu

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include "filter.h"

int clr_spc = 0;
void cllbck(int val, void* userdata) {
    clr_spc = val;
}

// Driver Function
int main(int argc, char* argv[]) {
    cv::VideoCapture* vid;


    // open the video device
    vid = new cv::VideoCapture(0);
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



    // Printing the Key Options
    printf("-------------Controls-------------\n g -> Grayscale\n h -> Alternative Grayscale\n b -> Gaussian Blur\n x -> Sobel X\n y -> Sobel Y\n m -> Magnitude Gradient\n l -> Blur and Quantize\n c -> Cartoonize\n n -> Negative\n e -> (Extension 1)Half Negative\n w -> (Extension 2)Colour Space Slider\n w -> (Extension 3)Colour Space Slider\n r -> Brightness Reduction\n p -> (Extension 4)Save modified frame\n t -> (Extension 5)Add a caption to current frame\n q -> Quit\n----------------------------------\n");

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
        if (key != -1){ // Only enters loop if a kry is pressed and resets some flags
            filt = key;
            bar_flag = 0;
            cv::destroyAllWindows();
        }
        
        // switch case to check Key pressed and display output
        switch (key_check(frame, dest, filt)) {
        
        case -1: break;    // if no key is pressed or if keys "g,h,b,x,y,m,l,c,n,e" are pressed
                           // function calls happen in the key_check function

        case 1: cv::namedWindow("Colour Space", cv::WINDOW_AUTOSIZE);  // if key 'w' is pressed ----------- it was hard to implement the slider in filter.cpp
            
            if (!bar_flag) { // this if condition ensures that the slider will not be reinitializzed till the next time the same key is pressed
                cv::createTrackbar("C_Space", "Colour Space", NULL, 4, cllbck);
                bar_flag = 1;  
            } 
            extension_2(frame, dest, clr_spc); // function call for (extension 2) color space changer
            cv::imshow("Extension 2", dest);
            break;

        case 9: cv::namedWindow("Reduce Brightness", cv::WINDOW_AUTOSIZE);  // if key 'r' is pressed ----------- it was hard to implement the slider in filter.cpp

            if (!bar_flag) { // this if condition ensures that the slider will not be reinitializzed till the next time the same key is pressed
                cv::createTrackbar("Dim by", "Reduce Brightness", NULL, 10, cllbck);
                bar_flag = 1;
            }
            extension_3(frame, dest, clr_spc); // function call for (extension 3) light reduction
            cv::imshow("Extension 3", dest);
            break;

        case 2: cv::destroyAllWindows();  // if key 'q' is pressed ----- Stop runtime and close all windows
            return 0;

        case 3:cv::imwrite("Saved_Frame.jpg", frame); // if key 's' is pressed ----- save current frame from webcam
            break;

        case 4:
            if(dest.empty()) {      //checks for filter output, if none then breaks
                printf("No filter output.\n");
                break;
            }
            cv::imwrite("Modified_Frame.jpg", dest); // if key 'p' is pressed ----- save current frame from real time filter output
            break;

        case 5:
            if (dest.empty()) {     //checks for filter output, if none then breaks
                printf("No filter output.\n");
                break;
            }
            printf("\n\nWhat would you like as a caption for the captured image: "); // if key 't' is pressed ----- caption the real time output
            getline(std::cin, caption);
            cv::putText(dest, caption, cv::Point(20, 400), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv::LINE_AA);
            cv::imshow("Extension 3 = Captioned Piece", dest);
            filt = -1;
            break;
        }

    }   // end of switch case
    delete vid;
    return(0);
}


// End of program