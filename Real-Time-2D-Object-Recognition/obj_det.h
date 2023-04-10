// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 2: Real time 2d Object Detection 

//obj_det.h
// custom created function declarations

// function defining comments inside function blocks in function definitions

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include "csv_util.h"



int key_check(cv::Mat& src, cv::Mat& dst, char key, std::vector<cv::Scalar>& colors);
int gaussian_blur5x5(cv::Mat& src, cv::Mat& dst);

int threshold(cv::Mat& src, cv::Mat& dst);
//int clean_up(cv::Mat& src, int num_rounds);
int clean_up(cv::Mat& bin_img);
//int dilation_erosion(cv::Mat src, int process_key);

int connected_component_analysis(cv::Mat& src, cv::Mat& dst);

int rand_color_generator(std::vector<cv::Scalar>& colors);
int segmentation(cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::Scalar> colors);

std::vector<cv::RotatedRect> bounding_boxes(cv::Mat& frame, cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map);
cv::Mat find_moments(cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::RotatedRect>& box);


int ft_vec(cv::Mat& frame, cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::RotatedRect>& box, int ctr, int classifier);

std::string nearest_neighbour(std::vector<float> ft_vec);
std::string k_nearest_neighbour(std::vector<float> ft_vec, int k);