// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 1:  Real-time filtering 


//filter.cpp -- file for storing function definitions to be referred in vidDisplay

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include <cmath>
#include "filter.h"

//Alternative Grayscale////////////////////////////////////////////////////////////////////////////
int greyscale(cv::Mat& src, cv::Mat& dst) {
	
	dst = cv::Mat::zeros(src.size(), CV_8UC3); // unsigned short 

	//for each row
	for (int i = 0; i < src.rows; i++) {
		int avg = 0;
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
		//for each column
		for (int j = 0; j < src.cols; j++) {
			avg = (rptr[j][0] + rptr[j][1] + rptr[j][2]) / 3;
			dptr[j][0] = avg;										//assign average value of all channel magnitudes
			dptr[j][1] = avg;
			dptr[j][2] = avg;
		}
	}
	return 0;
}
//EOF


//Gaussian Blur////////////////////////////////////////////////////////////////////////////////////
int blur5x5(cv::Mat& src, cv::Mat& dst) {
	// Gaussian blur filter
	//[ 1 2 4 2 1]
	dst = cv::Mat::zeros(src.size(), CV_16SC3);
	
	// gaussian horizontal filter = { 1, 2, 4, 2, 1 };

	for (int i = 0; i < src.rows; i++) {	      // for each row
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);  // src pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);  // destination pointer
		
		for (int j = 2; j < src.cols - 2; j++) {  // for each column
			
			for (int c = 0; c < 3; c++) {         // for each color channel
				dptr[j][c] = ( 1 * rptr[j - 2][c] + 2 * rptr[j -1][c] + 4 * rptr[j][c] + 2 * rptr[j + 1][c] + 1 * rptr[j + 2][c]) / 10;
			}
		}
	}
	
	cv::Mat temp = cv::Mat::zeros(dst.size(), CV_16SC3);
	dst.copyTo(temp);

	// gaussian vertical filter = { 1; 2; 4; 2; 1 };
	//for each row
	for (int i = 2; i < src.rows - 2; i++) {
		// src pointer
		cv::Vec3s* rptrp2 = temp.ptr<cv::Vec3s>(i - 2);
		cv::Vec3s* rptrp1 = temp.ptr<cv::Vec3s>(i - 1);
		cv::Vec3s* rptr0 = temp.ptr<cv::Vec3s>(i);
		cv::Vec3s* rptra1 = temp.ptr<cv::Vec3s>(i + 1);
		cv::Vec3s* rptra2 = temp.ptr<cv::Vec3s>(i + 2);
		// destination pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		// for each column
		for (int j = 0; j < src.cols; j++) {
			// for each color channel
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = (1 * rptrp2[j][c] + 2 * rptrp1[j][c] + 4 * rptr0[j][c] + 2 * rptra1[j][c] + 1 * rptra2[j][c]) / 10;
			}
		}
	}
	return 0;
} 
// EOF

//Sobel X filter for horizontal gradient///////////////////////////////////////////////////////////
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
	//sobel x filter
	dst = cv::Mat::zeros(src.size(), CV_16SC3);

	// vertical filter = { 1; 2; 1 };
	for (int i = 1; i < src.rows - 1; i++) {
		// src pointer
		cv::Vec3b* rptrp1 = src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* rptr0 = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* rptra1 = src.ptr<cv::Vec3b>(i + 1);
		// destination pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		// for each column
		for (int j = 0; j < src.cols; j++) {
			// for each color channel
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = (1 * rptrp1[j][c] + 2 * rptr0[j][c] + 1 * rptra1[j][c]) / 4;
			}
		}
	}

	// Horizontal { -1 0 1 };
	cv::Mat temp = cv::Mat::zeros(dst.size(), CV_16SC3);
	dst.copyTo(temp);
	for (int i = 0; i < temp.rows; i++) {
		// src pointer
		cv::Vec3s* rptr = temp.ptr<cv::Vec3s>(i);
		// destination pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		// for each column
		for (int j = 1; j < src.cols - 1; j++) {
			// for each color channel
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = (-1 * rptr[j - 1][c] + 0 * rptr[j][c] + 1 * rptr[j + 1][c]);
			}
		}
	}

	return 0;
}
//EOF


//Sobel Y for vertical gradient////////////////////////////////////////////////////////////////////
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
	//sobel y filter
	dst = cv::Mat::zeros(src.size(), CV_16SC3);

	// vertical filter = { 1; 0; -1 }
	for (int i = 1; i < src.rows - 1; i++) {
		// src pointer
		cv::Vec3b* rptrp1 = src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* rptr0 = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* rptra1 = src.ptr<cv::Vec3b>(i + 1);
		// destination pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		// for each column
		for (int j = 0; j < src.cols; j++) {
			// for each color channel
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = (1 * rptrp1[j][c] + 0 * rptr0[j][c] - 1 * rptra1[j][c]) ;
			}
		}
	}

	// horizontal { 1 2 1 };
	cv::Mat temp = cv::Mat::zeros(dst.size(), CV_16SC3);
	dst.copyTo(temp);
	for (int i = 0; i < temp.rows; i++) {
		// src pointer
		cv::Vec3s* rptr = temp.ptr<cv::Vec3s>(i);
		// destination pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		// for each column
		for (int j = 1; j < src.cols - 1; j++) {
			// for each color channel
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = (1 * rptr[j - 1][c] + 2 * rptr[j][c] + 1 * rptr[j + 1][c])/4;
			}
		}
	}

	return 0;
}
//EOF


//Magnitude Gradient///////////////////////////////////////////////////////////////////////////////
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
	dst = cv::Mat::zeros(sx.size(), CV_16SC3); // signed short 
	int mag;
	//for each row
	for (int i = 0; i < sy.rows; i++) {
		//  sobel x and sobel y output pointers
		cv::Vec3s* px = sx.ptr<cv::Vec3s>(i);
		cv::Vec3s* py = sy.ptr<cv::Vec3s>(i);
		// destination pointers
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		//for each column
		for (int j = 0; j < sy.cols; j++) {
			//for each color channel
			for (int c = 0; c < 3; c++) {
				mag = sqrt(pow(px[j][c], 2) + pow(py[j][c], 2)); //  combining sobel x and sobel y gradients using Euclidean distance
				dptr[j][c] = mag;
			}
		}
	}
	return 0;
}
//EOF


//Blur and Quantize////////////////////////////////////////////////////////////////////////////////
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
	//blurring
	blur5x5(src, dst);
	cv::Mat temp = cv::Mat::zeros(dst.size(), CV_8UC3);
	dst.copyTo(temp);
	//quantizing
	int b, xt, xf;
	b = 255 / levels;

	//for each row
	for (int i = 0; i < temp.rows; i++) {
		cv::Vec3s* sptr = temp.ptr<cv::Vec3s>(i);
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		//for each column
		for (int j = 0; j < temp.cols; j++) {
			//for each color channel
			for (int c = 0; c < 3; c++) {
				xt = sptr[j][c] / b;                         // dividing value by level
				xf = xt * b;								// fitting into bucket
				dptr[j][c] = xf;							//store to destination
			}
		}
	}
	return 0;
}//EOF



//Cartoonize///////////////////////////////////////////////////////////////////////////////////////
int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
	dst = cv::Mat::zeros(src.size(), CV_16UC3); // unsigned short
	//gradient magnitude
	cv::Mat sx, sy, temp;
	sobelX3x3(src, sx);
	sobelY3x3(src, sy);
	magnitude(sx, sy, temp);
	//blurring
	blurQuantize(src, dst, 15);
	
	//thresholding
	int flag = 0;
	for (int i = 0; i < temp.rows; i++) {
		cv::Vec3s* sptr = temp.ptr<cv::Vec3s>(i);
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		for (int j = 0; j < temp.cols; j++) {
			//for each color channel
			flag = 0;
			for (int c = 0; c < 3; c++) {
				if (sptr[j][c] > magThreshold) {  //if any channel has value greater than threshold inc flag
					flag++;
				}
			}
			// If any channel has value greater than threshold, then blacken entire pixel
			if (flag != 0) {
				dptr[j][0] = 0;
				dptr[j][1] = 0;
				dptr[j][2] = 0;
			}
		}
	}
	return 0;
}
//EOF



//Negative----(Part 10)////////////////////////////////////////////////////////////////////////////
int negative(cv::Mat& src, cv::Mat& dst) {
	dst = cv::Mat::zeros(src.size(), CV_8UC3); // unsigned short 

	for (int i = 0; i < src.rows; i++) {

		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {
			dptr[j][0] = 255 - rptr[j][0];			// Complement all channels without loop
			dptr[j][1] = 255 - rptr[j][1];
			dptr[j][2] = 255 - rptr[j][2];
		}
	}
	return 0;
}
//EOF


//    (Extenision 1)----Semi Negative     /////////////////////////////////////////////////////////
int extension_1(cv::Mat& src, cv::Mat& dst) {
	dst = cv::Mat::zeros(src.size(), CV_8UC3); // unsigned short 

	for (int i = 0; i < src.rows; i++) {

		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {
			if(j<src.cols/2){										//   For half the colums, invert
			dptr[j][0] = 255 - rptr[j][0];
			dptr[j][1] = 255 - rptr[j][1];
			dptr[j][2] = 255 - rptr[j][2];}
			else {													// The other half, keep as it is
				dptr[j][0] = rptr[j][0];
				dptr[j][1] = rptr[j][1];
				dptr[j][2] = rptr[j][2];
			}
		}
	}

	//Caption both sides of the window
	cv::putText(dst, "Negative", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv::LINE_AA);
	cv::putText(dst, "Original", cv::Point(640-120, 40), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv::LINE_AA);
	return 0;
}
//EOF



//    (Extenision 2)----Colour Space changer     //////////////////////////////////////////////////
int extension_2(cv::Mat& src, cv::Mat& dst, int bar) {
	dst = cv::Mat::zeros(src.size(), CV_8UC3); // unsigned short 

	//Depending on the slider value use cvt color function to change colour space
	if (bar == 4) {
		cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	}
	if (bar == 1) {
		cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	}
	if (bar == 2) {
		cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);
	}
	if (bar == 3) {
		cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
	}

	// Caption for Colour spaces assigned to each slider value
	cv::putText(dst, "1 => RGB", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv::LINE_AA);
	cv::putText(dst, "2 => HSV", cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv::LINE_AA);
	cv::putText(dst, "3 => Lab", cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv::LINE_AA);
	cv::putText(dst, "4 => Gray", cv::Point(10, 145), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv::LINE_AA);
	return 0;
}
//EOF


//    (Extenision 3)----Slider to dim the image  //////////////////////////////////////////////////
int extension_3(cv::Mat& src, cv::Mat& dst, int bar) {
	dst = cv::Mat::zeros(src.size(), CV_8UC3); // unsigned short 
	bar = bar+1;

	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {
			for (int c = 0; c < 3; c++) {
				if (bar==11) 
					dptr[j][c] = 0;							// If slider is at zero, then blacken the image
				else
					dptr[j][c] = rptr[j][c]/(bar);			// Reduce pixel channel magnitudes by scaling against the slider value
			}
		}
	}

	return 0;
}
//EOF

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    Key check function to check the value of key pressed and carry out related function call and operations    //////////////
cv::Mat sx, sy;
int key_check(cv::Mat& frame, cv::Mat& dest, char filt) {

	switch (filt) {
	case -1: break;
	case 'g':
	case 'G': cv::cvtColor(frame, dest, cv::COLOR_BGR2GRAY);
		cv::imshow("Gray", dest);
		return -1;
		break;
	case 'h':
	case 'H': greyscale(frame, dest);
		cv::imshow("Gray", dest);
		return -1;
		break;
	case 'b':
	case 'B': blur5x5(frame, dest);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Gaussian Blur", dest);
		return -1;
		break;
	case 'x':
	case 'X': sobelX3x3(frame, dest);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Sobel X", dest);
		return -1;
		break;
	case 'y':
	case 'Y': sobelY3x3(frame, dest);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Sobel Y", dest);
		return -1;
		break;
	case 'm':
	case 'M':sobelX3x3(frame, sx);
		sobelY3x3(frame, sy);
		magnitude(sx, sy, dest);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Magnitude", dest);
		return -1;
	case 'l':
	case 'L': blurQuantize(frame, dest, 15);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Blur and Quantize", dest);
		return -1;
	case 'c':
	case 'C': cartoon(frame, dest, 15, 15);
		cv::convertScaleAbs(dest, dest);
		cv::imshow("Cartoonized", dest);
		return -1;
	case 'n':
	case 'N': negative(frame, dest);
		cv::imshow("Negative", dest);
		return -1;
	case 'e':
	case 'E': extension_1(frame, dest);
		cv::imshow("Extension 1 = Semi_Negative", dest);
		return -1;
	case 'w':
	case 'W': return 1;
		
	case 'q':
	case 'Q': cv::destroyAllWindows();
		return 2;
	case 's':
	case 'S': return 3;
	case 'p':
	case 'P': return 4;
	case 't':
	case 'T': return 5;
	case 'r':
	case 'R': return 9;
	}
}
//EOF



////////  End of filter.cpp ///////////////////////////////////////////////////////////////////////////////////