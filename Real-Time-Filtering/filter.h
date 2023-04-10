// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 1:  Real-time filtering 


//filter.h -- header file for storing function prototypes to be referred in vidDisplay

#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

int greyscale(cv::Mat& src, cv::Mat& dst);

int blur5x5(cv::Mat& src, cv::Mat& dst);

int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);

int negative(cv::Mat& src, cv::Mat& dst);

int extension_1(cv::Mat& src, cv::Mat& dst);
int extension_2(cv::Mat& src, cv::Mat& dst, int bar);
int extension_3(cv::Mat& src, cv::Mat& dst, int bar);

int key_check(cv::Mat& src, cv::Mat& dst, char key);