// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// //PRJ 2
// CBIR.cpp : This file contains the 'main' function. Program execution begins and ends there.


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <dirent.h>
#include "Feaure_Saver.h"

int main(int argc, char** argv)
{
	char key = -1;

	// Read the image file
	cv::Mat img = cv::imread("olympus/olympus/pic.0274.jpg");

	// Check for failure
	if (img.empty())
	{
		std::cout << "Image Not Found!!!" << std::endl;
		std::cin.get(); //wait for any key press
		return -1;
	}

	// Show the target image inside a window.
	cv::imshow("Target Image", img);
	std::cout << "\n ------Menu------\n"
		<< "1. Baseline Matching\n" << "2. Single Histogram Intersection"
		<< "\n3. Multi-Histogram Matching" << "\n4. Texture and Color"
		<< "\n5. Custom Design(Texture with centralized histogram)";
	printf("\n\n Enter option: ");
	key = cv::waitKey(0);
	char c = key;
	printf(" %c",c);
	mainline(img, c);
	// Wait for any keystroke in the window
	char k = 0;
	while (1) {
		k = cv::waitKey(0);
		if (k == 'q')
		{
			cv::destroyAllWindows();
			exit(1);
		}
		else
			cv::waitKey(20);
	}
	return 0;
}
