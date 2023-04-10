// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// //PRJ 2


// This is the declaration file for the custom functions used. 
// The function description comments are included in the function definitions in the cppp file.

#pragma once


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <string>
#include <vector>
#include <iostream>
#include <dirent.h>
#include "Feaure_Saver.h"


float intersection_dist(std::vector<float>& ha, std::vector<float>& hb);

int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

int single_hist(cv::Mat src, std::vector<float>& ft);

int multi_hist(cv::Mat& src, std::vector<float>& ft, std::vector<float>& ft2);

int texture(cv::Mat& src, std::vector<float>& ft);

int custom_red_ball(cv::Mat& src, std::vector<float>& ft, std::vector<float>& ft2);

int bm_ft_gen(cv::Mat& src, std::vector<float>& ft);
int bm_dist(std::vector<float> test_ft, std::vector<float> target_ft);

int mainline(cv::Mat& src, char key);