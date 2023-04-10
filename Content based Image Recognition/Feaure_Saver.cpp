// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// //PRJ 2

// This file contains the different custom functions definitions used.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <string>
#include <vector>
#include <iostream>
#include <dirent.h>
#include "Feaure_Saver.h"


///////////////////////Histograms/////////////////////////////////
/*
Functions used in histogram intersections. Individual descriptions inside funcction blocks.
*/
int single_hist(cv::Mat src, std::vector<float>& ft) {
	/*
	A histogram is computed from the 3 channels of the input image and then flattened into a vector. 
	This vector is also an input by reference. The feature vector is then normalised and the function returns zero.

	Input : Image Mat, Feature vector by reference
	Output : Vector called by reference.
	*/

	const int Hsize = 8;
	const int divisor = 256 / Hsize;
	int i, j;
	int dim[3] = { Hsize, Hsize, Hsize };

	//allocate and intialize a histogram
	cv::Mat hist(3, dim, CV_32S, cv::Scalar(0));


	for (i = 0; i < src.rows; i++) {
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		for (j = 0; j < src.cols; j++) {
			int r = sptr[j][2] / divisor; //R index
			int g = sptr[j][1] / divisor; //G index
			int b = sptr[j][0] / divisor; //B index
			//printf("%d", r);
			hist.at<int>(r, g, b)++; // increment the histogram at location (r,g,b)
		}
	}

	// hist is an integer histogram
	// not a normalised histogram

	//Flattening Mat objects
	for (int k = 0; k < Hsize; k++) {
		for (int i = 0; i < Hsize; i++) {
			for (int j = 0; j < Hsize; j++) {
				ft.push_back(hist.at<int>(k, j, i));
			}
		}
	}
	//Normalize vector here and store instead of in distance calc to avoid rep
	float suma = 0.0;
	for (int i = 0; i < ft.size(); i++) {
		suma += ft[i];						// Sum of hist buckets
	}
	for (int i = 0; i < ft.size(); i++) {
		ft.at(i) /= suma;					// normalizing
	}
	return 0;
}

int multi_hist(cv::Mat& src, std::vector<float>& ft, std::vector<float>& ft2) {
	/*
	A histogram is computed from the top and bottom halves of the input image and then flattened into 2 vectors.
	These vectors are also inputs by reference. The feature vectors are then normalised and the function returns zero.

	Input : Image Mat, 2 Feature vector by reference for top and half
	Output : Vector called by reference. otherwise zero
	*/
	const int Hsize = 8;
	const int divisor = 256 / Hsize;
	int i, j;
	int dim[3] = { Hsize, Hsize, Hsize };

	//allocate and intialize a histogram
	cv::Mat hist(3, dim, CV_32S, cv::Scalar(0));

	//top half
	for (i = 0; i < src.rows/2; i++) {
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		for (j = 0; j < src.cols; j++) {
			int r = sptr[j][2] / divisor; //R index
			int g = sptr[j][1] / divisor; //G index
			int b = sptr[j][0] / divisor; //B index
			//printf("%d", r);
			hist.at<int>(r, g, b)++; // increment the histogram at location (r,g,b)
		}
	}
	//Flattening Mat object for top half
	for (int k = 0; k < Hsize; k++) {
		for (int i = 0; i < Hsize; i++) {
			for (int j = 0; j < Hsize; j++) {
				ft.push_back(hist.at<int>(k, j, i));
			}
		}
	}

	//bottom half
	cv::Mat hist2(3, dim, CV_32S, cv::Scalar(0));
	for (i = 0; i < src.rows / 2; i++) {
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		for (j = 0; j < src.cols; j++) {
			int r = sptr[j][2] / divisor; //R index
			int g = sptr[j][1] / divisor; //G index
			int b = sptr[j][0] / divisor; //B index
			//printf("%d", r);
			hist2.at<int>(r, g, b)++; // increment the histogram at location (r,g,b)
		}
	}
	//Flattening Mat object for top half
	for (int k = 0; k < Hsize; k++) {
		for (int i = 0; i < Hsize; i++) {
			for (int j = 0; j < Hsize; j++) {
				ft2.push_back(hist2.at<int>(k, j, i));
			}
		}
	}

	//Normalize vector here and store instead of in distance calc to avoid rep
	float suma = 0.0;
	float sumb = 0.0;
	for (int i = 0; i < ft.size(); i++) {
		suma += ft[i];						// Sum of hist buckets
		sumb += ft2[i];						// Sum of hist buckets
	}
	for (int i = 0; i < ft.size(); i++) {
		ft.at(i) /= suma;					// normalizing
		ft2.at(i) /= sumb;
	}
	return 0;
}
 
int texture(cv::Mat& src, std::vector<float>& ft){
	/*
	A histogram is computed from the gradient magnitudes of the input image. This is done by using the sobel x and y functions. 
	The magnitude function then uses euclidean distance to combine the sobel x and y outputs to give a gradient magnitude mat 
	from which a histogram is calculated. The histogram is then flattened into a vector and normalized. These vectors are also inputs by reference.
	
	Input : Image Mat, Feature vector by reference
	Output : Vector called by reference. otherwise zero
	*/

	// Getting the gradient magnitude
	cv::Mat dest, sx, sy;
	sobelX3x3(src, sx);					// sobel x
	sobelY3x3(src, sy);					// sobel y
	magnitude(sx, sy, dest);			// magnitude
	cv::convertScaleAbs(dest, dest);	
	/*cv::imshow("Magnitude", dest);
	cv::waitKey(0);*/


	//getting a histogram out
	const int Hsize = 8;
	const int divisor = 256 / Hsize;
	int i, j;
	int dim[3] = { Hsize, Hsize, Hsize };

	//allocate and intialize a histogram
	cv::Mat hist(3, dim, CV_32S, cv::Scalar(0));


	for (i = 0; i < src.rows; i++) {
		cv::Vec3b* sptr = dest.ptr<cv::Vec3b>(i);
		for (j = 0; j < src.cols; j++) {
			int r = sptr[j][2] / divisor; //R index
			int g = sptr[j][1] / divisor; //G index
			int b = sptr[j][0] / divisor; //B index
			//printf("%d", r);
			hist.at<int>(r, g, b)++; // increment the histogram at location (r,g,b)
		}
	}

	// hist is an integer histogram
	// not a normalised histogram

	//Flattening Mat object
	for (int k = 0; k < Hsize; k++) {
		for (int i = 0; i < Hsize; i++) {
			for (int j = 0; j < Hsize; j++) {
				ft.push_back(hist.at<int>(k, j, i));
			}
		}
	}
	//Normalize vector here and store instead of in distance calc to avoid rep
	float suma = 0.0;
	for (int i = 0; i < ft.size(); i++) {
		suma += ft[i];						// Sum of hist buckets
	}
	for (int i = 0; i < ft.size(); i++) {
		ft.at(i) /= suma;					// normalizing
	}
	return 0;
}

float intersection_dist(std::vector<float>& ha, std::vector<float>& hb) {
	double intersection = 0.0;
	double af = 0.0, bf = 0.0;

	//compute intersection          
	for (int i = 0; i < ha.size(); i++) {
		intersection += ha[i] < hb[i] ? ha[i] : hb[i]; // shorthand tripartite assignment
	}

	// convert to a distance
	return (1.0 - intersection);
}


int custom_red_ball(cv::Mat& src, std::vector<float>& ft, std::vector<float>& ft2) {

	//hist matching of a 16x16 roc
	int start_y = (src.rows / 2) - 23;
	int start_x = (src.cols / 2) - 23;

	const int Hsize = 8;
	const int divisor = 256 / Hsize;
	int i, j;
	int dim[3] = { Hsize, Hsize, Hsize };

	//allocate and intialize a histogram
	cv::Mat hist(3, dim, CV_32S, cv::Scalar(0));

	for (i = 0; i < 16 * 4; i++) {
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(start_y + i);
		for (j = 0; j < 16 * 4; j++) {
			int r = sptr[start_x + j][2] / divisor; //R index
			int g = sptr[start_x + j][1] / divisor; //G index
			int b = sptr[start_x + j][0] / divisor; //B index
			hist.at<int>(r, g, b)++; // increment the histogram at location (r,g,b)
		}
	}

	//Flattening Mat objects
	for (int k = 0; k < Hsize; k++) {
		for (int i = 0; i < Hsize; i++) {
			for (int j = 0; j < Hsize; j++) {
				ft2.push_back(hist.at<int>(k, j, i));
			}
		}
	}
	//Normalize vector here and store instead of in distance calc to avoid rep
	float suma = 0.0;
	for (int i = 0; i < ft2.size(); i++) {
		suma += ft2[i];						// Sum of hist buckets
	}
	for (int i = 0; i < ft2.size(); i++) {
		ft2.at(i) /= suma;					// normalizing
	}


	// Getting the gradient magnitude
	texture(src, ft);
	return 0;
}

///////////////////////Dependencies/////////////////////////////////
/*
These are functions from the previous project. Used for texture feature vector calculations.
*/
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
				dptr[j][c] = (1 * rptrp1[j][c] + 0 * rptr0[j][c] - 1 * rptra1[j][c]);
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
				dptr[j][c] = (1 * rptr[j - 1][c] + 2 * rptr[j][c] + 1 * rptr[j + 1][c]) / 4;
			}
		}
	}

	return 0;
}
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



///////////////////////Baseline MATCHING/////////////////////////////////
/*
These are the two functions used in baseline matching. Individual descriptions are in commments in the block.
*/
int bm_ft_gen(cv::Mat& src, std::vector<float>& ft) {
	/*
	A feature vector genration function for baseline matching. Calculates the midpoint of the image and then 
	stores the values from a 9x9 frame around it; into a vector that is called by reference.

	Input : Image Mat, Feature vector by reference
	Output : Vector called by reference. otherwise zero
	*/
	// Step 1: Obtaining feature vector

	//cv::Mat BL = cv::Mat::zeros(3, 3, CV_8UC1);
	int mid_r, mid_c;
	mid_r = src.rows / 2;
	mid_c = src.cols / 2;
	//printf("\n\nMidpoint is: %d , %d", mid_r, mid_c);
	//printf("\n\nOrig_dimensions are: %d , %d", mid_r * 2, mid_c * 2);
	int sx = mid_c - 4;
	int sy = mid_r - 4;

	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < 9; i++) {
			cv::Vec3b* sptr = src.ptr<cv::Vec3b>(sy + i);
			for (int j = 0; j < 9; j++) {
				ft.push_back(sptr[sx + j][k]);
			}
		}
	}
	return(0);
}
int bm_dist(std::vector<float> test_ft, std::vector<float> target_ft) {
	/*
	Computes the sum of squared distances from input feature vectors and returns it.

	Input : Feature vectors
	Output : distance
	*/
	int dist = 0, diff = 0;
	for (int i = 0; i < test_ft.size(); i++) {
		diff = test_ft[i] - target_ft[i];
		dist += (diff * diff);
	}

	return dist;
}


///////////////////////Main Pipeline/////////////////////////////////
/*
This function serves as the main pipeline for the program. 
It takes in the target image and the option selected from the main function and uses it in a switchcase to 
iteratively pass test cases to the required feature vector calculation function.

Input : Target Image, Option Selected (Char from keyboard)
*/
int mainline(cv::Mat& src, char key) {
	//readfiles.cpp
	char dirname[256];
	char buffer[256];
	int i = 1;
	float w = 0.0;
	FILE* fp = nullptr;
	DIR* dirp;
	struct dirent* dp;
	cv::Mat test;
	std::vector<std::pair <float, std::string> > temp;
	std::vector<float> ft, ft2, ft_target, ft_target2;
	if (key == '1') {
		bm_ft_gen(src, ft_target);
	}
	if (key == '2') {
		single_hist(src, ft_target);
	}
	if (key == '3') {
		multi_hist(src, ft_target, ft_target2);
		printf("How do you want the top half to be weighed in a scale of 0 to 1 (bottom half will be 1-weight): ");
		std::cin >> w;
	}
	if (key == '4') {
		single_hist(src, ft_target);
		texture(src, ft_target2);
	}
	if (key == '5') {
		custom_red_ball(src, ft_target, ft_target2);

	}

	strcpy(dirname, "olympus/olympus/");
	printf("Processing directory %s\n", dirname);

	// open the directory
	dirp = opendir(dirname);
	if (dirp == NULL) {
		printf("Cannot open directory %s\n", dirname);
		exit(-1);
	}

	// loop over all the files in the image file listing
	while ((dp = readdir(dirp)) != NULL) {
		//clearing the vector
		ft.vector::clear();
		ft2.vector::clear();
		// check if the file is an image
		if (strstr(dp->d_name, ".jpg") ||
			strstr(dp->d_name, ".png") ||
			strstr(dp->d_name, ".ppm") ||
			strstr(dp->d_name, ".tif")) {

			//printf("processing image file: %s\n", dp->d_name);

			// build the overall filename
			strcpy(buffer, dirname);
			//strcat(buffer, "/");
			strcat(buffer, dp->d_name);

			//printf("full path name: %s\n", buffer);

			// Different tasks

			test = cv::imread(buffer);

			if (key == '1') {
				bm_ft_gen(test, ft);
				temp.push_back(std::make_pair(bm_dist(ft, ft_target), buffer));
			}
			if (key == '2') {
				single_hist(test, ft);
				float dit = intersection_dist(ft, ft_target);
				temp.push_back(std::make_pair(dit, buffer));
			}
			if (key == '3') {
				multi_hist(test, ft, ft2);
				float dit = intersection_dist(ft, ft_target);
				float dit2 = intersection_dist(ft2, ft_target2);
				temp.push_back(std::make_pair((w * dit + (1 - w) * dit2), buffer));
			}
			if (key == '4') {
				single_hist(test, ft);
				texture(test, ft2);
				float dit = intersection_dist(ft, ft_target);
				float dit2 = intersection_dist(ft2, ft_target2);
				temp.push_back(std::make_pair((dit + dit2), buffer));
			}
			if (key == '5') {
				custom_red_ball(test, ft, ft2);
				float dit = intersection_dist(ft, ft_target);
				float dit2 = intersection_dist(ft2, ft_target2);
				w = 0.6;
				temp.push_back(std::make_pair((w * dit + (1 - w) * dit2), buffer));
			}
			i++;
		}
	}

	std::sort(temp.begin(), temp.end());
	std::cout << "\n" << temp[0].first << "\n" << temp[0].second;
	std::cout << "\n" << temp[1].first << "\n" << temp[1].second;
	std::cout << "\n" << temp[2].first << "\n" << temp[2].second;
	std::cout << "\n" << temp[3].first << "\n" << temp[3].second;
	std::cout << "\n" << temp[4].first << "\n" << temp[4].second;

	test = cv::imread(temp[0].second);

	cv::Size size(320, 240);
	cv::resize(test, test, size);
	cv::imshow("Best Match", test);

	for (int i = 1; i <= 9; i++) {
		cv::Mat img = cv::imread(temp[i].second, cv::IMREAD_COLOR);

		cv::resize(img, img, size);
		cv::imshow("Match_" + std::to_string(i + 1), img);
	}

	return 0;
}