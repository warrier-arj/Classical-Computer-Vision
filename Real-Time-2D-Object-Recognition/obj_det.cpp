// Arjun Rajeev Warrier
// Spring 2023 CS 5330
// Project 2: Real time 2d Object Detection 

//obj_det.cpp
// custom created function definitions
// function defining comments inside function blocks


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <numeric>
#include "obj_det.h"
#include "csv_util.h"

std::vector<std::vector<cv::Point>> contours;
int classifier = 1;

int key_check(cv::Mat& frame, cv::Mat& dest, char filt, std::vector<cv::Scalar>& colors) {

	/*
	A pipeline for program. The main functin passes the frame from the feed and the user key press. 
	the functions are then called accordingly from a switch case

	Input : frame from camera feed, filt = key pressed, vector of random colours
	Output : returns an integer value that helps main decide to quit or continue
	*/

	cv::Mat bin_map, cleaned_bin_map, id_map, seg_map, frame_temp;
	frame_temp = frame.clone();
	std::vector<cv::RotatedRect> boxes;
	int ctr = 0;
	switch (filt) {
	case -1: break;
	case 'e':
	case 'E': classifier = 1;// neareset neighbour
		return -1;
		break;
	case 'w':
	case 'W': classifier = 2;// neareset neighbour
		return -1;
		break;
	case '1': threshold(frame, bin_map);
		cv::imshow("Thresholded", bin_map);
		//clean_up(bin_map, 2);
		//cv::imshow("Cleaned", dest);
		return -1;
		break;
		
	case '2':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		cv::imshow("Thresholded", bin_map);
		cv::imshow("Cleaned", cleaned_bin_map);
		return -1;
		break;

	case '3':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		segmentation(cleaned_bin_map, id_map, seg_map, colors);
		cv::imshow("Segmented", seg_map);
		return -1;
		break;
	case '4':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		segmentation(cleaned_bin_map, id_map, seg_map, colors);
		bounding_boxes(frame, bin_map, id_map, seg_map);
		cv::imshow("Bounding Boxes", seg_map);
		return -1;
		break;
	case '5':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		segmentation(cleaned_bin_map, id_map, seg_map, colors);
		boxes = bounding_boxes(frame, bin_map, id_map, seg_map);

		find_moments(bin_map, id_map, seg_map, boxes);
		cv::imshow("Bounding Boxes with moment axis", seg_map);
		return -1;
		break;
	case '6':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		segmentation(cleaned_bin_map, id_map, seg_map, colors);
		boxes = bounding_boxes(frame, bin_map, id_map, seg_map);
		ctr = 1;
		ft_vec(frame, bin_map, id_map, seg_map, boxes, ctr, classifier);
		cv::imshow("Saved_Image", seg_map);
		ctr = 0;
		return 3;
		break;
	case '7':threshold(frame, bin_map);
		cleaned_bin_map = bin_map.clone();
		clean_up(cleaned_bin_map);
		segmentation(cleaned_bin_map, id_map, seg_map, colors);
		boxes = bounding_boxes(frame, bin_map, id_map, seg_map);
		ft_vec(frame, bin_map, id_map, seg_map, boxes, ctr, classifier);
		cv::imshow("Match", frame);
		return 3;
		break;
	case 'q':
	case 'Q': return 2;
	}
}


/////////////////////////////////////////////////////////////////////////Task 1
int gaussian_blur5x5(cv::Mat& src, cv::Mat& dst) {
	// Gaussian blur filter
	//[ 1 2 4 2 1]
	dst = cv::Mat::zeros(src.size(), CV_16SC3);

	// gaussian horizontal filter = { 1, 2, 4, 2, 1 };

	for (int i = 0; i < src.rows; i++) {	      // for each row
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);  // src pointer
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);  // destination pointer

		for (int j = 2; j < src.cols - 2; j++) {  // for each column

			for (int c = 0; c < 3; c++) {         // for each color channel
				dptr[j][c] = (1 * rptr[j - 2][c] + 2 * rptr[j - 1][c] + 4 * rptr[j][c] + 2 * rptr[j + 1][c] + 1 * rptr[j + 2][c]) / 10;
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

int threshold(cv::Mat& src1, cv::Mat& dst) {
	/*
	It blurs and greyscales a frame from the video feed. 
	It then goes over each pixel in the region considered(edges have been ignored) and checks wheter the pixel is over the threshold.
	If yes, then it is a bg pixel = 0 otherwise it is a foreground pixel = 255.

	Input : 2 image Mats, one frame and one binary image
	Output : Binary image mat called by reference. Returns zero to terminate.
	*/
	cv::Mat src, temp;
	//gaussian_blur5x5(src1, temp);
	cv::blur(src1, temp, cv::Size(3, 3));
	greyscale(temp, src);
	//cv::convertScaleAbs(src, src);
	
	dst = cv::Mat::ones(src.size(), CV_8UC1);
	int val;

	for (int i = 0; i < src.rows; i++) {
		//cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		uchar* dptr = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			if (i <= 60 || i >= src.rows-60) {
				dptr[j] = 0;
			}
			else {
				dptr[j] = sptr[j][0] > 140 ? 0 : 255;
			}

		}
	}
	return 0;
}


/////////////////////////////////////////////////////////////////////////Task 2
//Clean_up functions... come back later
/*
int clean_up(cv::Mat& src, int num_rounds) {
	
	for (int i = 0; i < num_rounds; i++) {
		dilation_erosion(src, 0);
		dilation_erosion(src, 255);
	}
	return 0;
}

int dilation_erosion(cv::Mat src, int process_key) {
	cv::Mat dst1;
	dst1 = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == process_key)
			{
				dst1.at<uchar>(i + 1, j) = process_key;
				dst1.at<uchar>(i - 1, j) = process_key;
				dst1.at<uchar>(i, j + 1) = process_key;
				dst1.at<uchar>(i, j - 1) = process_key;
			}
		}
	}
	src = dst1.clone();
	return 0;
}
*/
int clean_up(cv::Mat& bin_img) {
	/*
	Takes in a binary image and performs dilation and erosion to clean up spots and holes in the thresholded image. 
	The function executes clean up and stores it to same Mat.

	Input : 1 image Mats, thresholded binary image
	Output : Binary image mat called by reference. Returns zero to terminate.
	*/
	cv::Mat temp;
	//printf("\n Function check 1")
	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));  //4-connected erosion
	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));   //8 connected dilation
	for (int i = 0; i < 3; i++) {
		cv::dilate(bin_img, temp, kernel2, cv::Point(-1, -1), 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		cv::erode(temp, bin_img, kernel1, cv::Point(-1, -1), 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	
	return 0;
}

/////////////////////////////////////////////////////////////////////////Task 3

int connected_component_analysis(cv::Mat& src, cv::Mat& dst) {
	/*
	2 pass algorithm for attempted segmentation.
	Does not work

	Input : 2 image Mats, one frame and one binary image
	Output : Binary image mat called by reference. Returns zero to terminate.
	*/
	int id = 0, max = 0;

	// First pass
	//for each row
	for (int i = 1; i < src.rows; i++) {
		uchar* sptr = src.ptr<uchar>(i);
		uchar* dptr = dst.ptr<uchar>(i);
		int* dptr_prev = dst.ptr<int>(i-1);

		//for each column
		for (int j = 1; j < src.cols; j++) {
			max = 0;
			if (sptr[j] != 0) {printf("\n\n Fi2424244ne");
				max = dptr[j-1];
				max = (dptr_prev[j - 1] > max) ? dptr_prev[j - 1] : max;
				max = (dptr_prev[j] > max) ? dptr_prev[j] : max;
				max = (dptr_prev[j + 1] > max) ? dptr_prev[j + 1] : max;
				if (max == 0) {
					printf("\n\n fffffffffffffffffffffffffffffff");
					id++;	
			}
			max = id;
			}
			
			dptr[j] = max;
		}
	}
	printf("\n\n First Pass done");
	// Second pass
	//for each row
	for (int i = src.rows - 1; i >= 0; i++) {
		uchar* sptr = src.ptr<uchar>(i);
		uchar* dptr = dst.ptr<uchar>(i);
		int* dptr_prev = dst.ptr<int>(i + 1);
		//for each column
		for (int j = src.cols - 1; j >= 0; j++) {
			max = 0;
			if (sptr[j] != 0) {
				max = dptr[j - 1];
				max = (dptr_prev[j - 1] > max) ? dptr_prev[j - 1] : max;
				max = (dptr_prev[j] > max) ? dptr_prev[j] : max;
				max = (dptr_prev[j + 1] > max) ? dptr_prev[j + 1] : max;
			}
			if (max == 0) {
				max = id;
				id--;
			}
			dptr[j] = max;
		}
	}
	return 0;
}

int rand_color_generator(std::vector<cv::Scalar>& colors) {
	/*
	Generates a set of 10 random colors and stores it to a vector for later usage.

	Input : A vector called by reference
	Output : Returns zero to terminate.
	*/
	// Initialize random seed
	srand((unsigned)time(NULL));

	// Generate 10 random colors
	for (int i = 0; i < 10; i++) {
		cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
		colors.push_back(color);
	}
	return 0;
}
int segmentation(cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::Scalar> colors) {
	/*
	Takes in a binary image. Segments it into different parts and gives each different ids.
	These segments are then stored in different colors (random color vector from prev function) in a mat called by ref.

	Input : binary image, id map and vector of different colors
	Output : segmented map of different colors callled by ref.
	*/

	//Using the connected component function from opencv
	cv::Mat stats, centroid;
	int num_segments = 0;
	num_segments = cv::connectedComponentsWithStats(bin_map, id_map, stats, centroid);
	
	//Colouring different segments in the map
	seg_map = cv::Mat::zeros(bin_map.size(), CV_8UC3);
	
	for (int i = 1; i < num_segments; i++) {
		if (stats.at<int>(i, cv::CC_STAT_AREA) > 75) {		// Only considering the region IDs that have area bigger than 75
			cv::Mat maskImg = id_map == i;
			seg_map.setTo(colors[i], maskImg);				// Assigning diff colors to diff segments
		}
	}
	return 0;
}


/////////////////////////////////////////////////////////////////////////Task 4,5

std::vector<cv::RotatedRect> bounding_boxes(cv::Mat& frame, cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map) {
	/*
	Draws rotation invariant bounding boxes around objects/segments

	Input : frame from video feed, binary image, id map and segmented map
	Output : bounding boxes drawn on frame and segmented map. The coordinates of bounding boxes are returned.
	*/


	//Using the connected component function from opencv
	cv::Mat stats, centroid;
	int num_segments = 0;
	num_segments = cv::connectedComponentsWithStats(bin_map, id_map, stats, centroid);

	std::vector<cv::Rect> boundingBoxes(num_segments - 1);
	std::vector<cv::RotatedRect> rotatedRect(num_segments - 1);
	for (int i = 1; i < num_segments; i++) {
		boundingBoxes[i - 1] = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
			stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
	}
	float min_area = 2000, max_area = 500000;

	contours.clear();
	cv::findContours(bin_map, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	int k = 0;
	for (auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > min_area && area < max_area) {			// restricting the segment sizes to be considered
			rotatedRect[k] = cv::minAreaRect(contour);		// a rotated rectangle encompassing all the segment pixels

			cv::Point2f rect_points[4];
			rotatedRect[k].points(rect_points);				// stores the vertices of rotated rectangle in rect_points

			
			//Drawing the bounding boxes
			for (int j = 0; j < 4; j++) {
				cv::line(frame, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 4);
			}
			for (int j = 0; j < 4; j++) {
				cv::line(seg_map, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(255, 255, 255), 4);
			}
		}
	}
	return rotatedRect;
}

cv::Mat find_moments(cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::RotatedRect>& box){
	
	/*
	Computes the moments of the segments in the binary image and draws axis.

	Input : binary image, id map, segmented map and coordinates of bounding boxes
	Output : returns a Mat with the moments
	*/

	//Calculating moments
	
	cv::Mat huMoments;
	std::vector<std::vector<cv::Point>> contour;
	
	int k = 0;
	for (auto& contour : contours) {
		cv::Moments moment = cv::moments(contour, false);
		
		cv::HuMoments(moment, huMoments);


		double mu20 = moment.mu20;
		double mu02 = moment.mu02;
		double mu11 = moment.mu11;

		float alpha;
		if (mu20 - mu02 == 0) {
			alpha = static_cast <float>(0);
		}
		else {
			alpha = static_cast<float>(atan2(2 * mu11, mu20 - mu02) / 2);
		}


		double xC = moment.m10 / moment.m00;
		double yC = moment.m01 / moment.m00;

		double rectHt = box[k].size.height;
		double rectWd = box[k].size.width;
		double len = rectHt > rectWd ? rectHt : rectWd;
		len *= 0.5;

		double major_x2 = xC + len * cos(alpha);
		double major_y2 = yC + len * sin(alpha);
		double minor_x2 = xC + len * -sin(alpha);
		double minor_y2 = yC + len * cos(alpha);

		cv::Point majorExtent, minorExtent;
		majorExtent.x = major_x2;
		majorExtent.y = major_y2;
		minorExtent.x = minor_x2;
		minorExtent.y = minor_y2;


		//Drawing moment axis
		cv::line(seg_map, cv::Point(xC, yC), majorExtent, cv::Scalar(0, 0, 255), 2);
		cv::line(seg_map, cv::Point(xC, yC), minorExtent, cv::Scalar(140, 130, 0), 2);
	}
	return huMoments;
}

int ft_vec(cv::Mat& frame, cv::Mat& bin_map, cv::Mat& id_map, cv::Mat& seg_map, std::vector<cv::RotatedRect>& box, int ctr, int classifier) {
	
	/*
	Computes the feature vectors: moments, aspect ratio, filling ratio
	If ctr = 1: store to csv file (database) named ft.csv
	If ctr = 0: call classifier function (based on value of int classifier) and print resulting name on bounding box in the frame
	
	Input : binary image, id map, segmented map and coordinates of bounding boxes, ctr for train/test, classifier option
	Output : return zero
	*/

	int k_means = 2;
	char filename[] = "ft.csv";
	char class_name[] = "";
	std::vector<float> ft_vec;
	double area, filled_area, asp_ratio, height, width;
	std::vector<std::string> matches;
	
	cv::findContours(bin_map, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat huMoments = find_moments(bin_map, id_map, seg_map, box);
	
	int k = 0;
	for (auto& contour : contours) {
		area = cv::contourArea(contour);
		height = box[k].size.height;
		width = box[k].size.width;
		//Adding moments to feature vectors
		for (int i = 0; i < huMoments.rows; i++) {
			ft_vec.push_back(huMoments.at<double>(i));
		}
		//adding aspect ration and area of the bounding box filled
		filled_area = area / static_cast<double>((height * width));
		asp_ratio = height / width;
		ft_vec.push_back(filled_area);
		ft_vec.push_back(asp_ratio);

		if (ctr)
		{   //class name
			printf("\n\nEnter object name: ");
			std::cin >> class_name;
			//add to db
			append_image_data_csv(filename, class_name, ft_vec, false);
		}
		else {
			if (classifier == 1)
				matches.push_back(nearest_neighbour(ft_vec));
			else
				matches.push_back(k_nearest_neighbour(ft_vec, k_means));
		}
		k++;
	}
	if(ctr){
		return 0;
	}
	else {
		int k = 0;
		for (auto& contour : contours){
			cv::putText(seg_map, matches[k], box[k].center, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
			cv::putText(frame, matches[k], box[k].center, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
			k++;
		}
	}
}


/////////////////////////////////////////////////////////////////////////Task 6,7
std::string nearest_neighbour(std::vector<float> ft_vec) {
	/*
	Nearest neighbour function to compute the scaled euclidean distances between target image and databse feature vectors
	database feature vectors are read from database and stored in a vector of vectors.
	Returns the best match

	Input : feature vector of image to be classified
	Output : return class name of best match
	*/

	// For standard deviations
	std::vector<float> stdDevs;  
	std::vector<std::pair<float, std::string>> matcher;

	// Getting the training data from csv
	char filename[] = "ft.csv";
	std::vector<char*> classes;
	std::vector<std::vector<float>> data;
	read_image_data_csv(filename, classes, data);

	//Computing standard deviations
	for (int i = 0; i < data.size(); i++) {
		float sum = 0,  mean = 0, var = 0;
		//Mean
		mean = std::accumulate(data[i].begin(), data[i].end(), 0.0) / data[i].size();
		//Variance
		for (const auto& val : data[i]) {
			var += (val - mean) * (val - mean);
		}
		var /= data[i].size();
		stdDevs.push_back(sqrt(var));
	}


	for (int i = 0; i < data.size(); i++) {

		float sum = 0;
		for (int j = 0; j < data[i].size(); j++) {
			sum += abs(ft_vec[j] - data[i][j]);
		}
		float dist = sqrt(sum) / stdDevs[i];
		matcher.push_back(std::make_pair(dist, classes[i]));

	}

	sort(matcher.begin(), matcher.end()); // sort on the basis of distances
	return matcher[0].second; //return name of closest match
}

std::string k_nearest_neighbour(std::vector<float> ft_vec, int k) {

	k = 2;

	// For standard deviations
	std::vector<float> stdDevs;
	std::vector<std::pair<float, std::string>> matcher;
	std::vector<double> distVec;

	// Getting the training data from csv
	char filename[] = "ft.csv";
	std::vector<char*> classes;
	std::vector<std::vector<float>> data;
	read_image_data_csv(filename, classes, data);



	std::set<std::string> st(classes.begin(), classes.end());

	for (const auto& currClass : st) {
		double dist = 0;
		for (int i = 0; i < data.size(); i++) {
			if (classes[i] == currClass) {
				double sum = 0;
				for (int j = 0; j < data[i].size(); j++) {
					sum += (ft_vec[j] - data[i][j]) * (ft_vec[j] - data[i][j]);

				}
				dist += sqrt(sum) / data[i].size();
				distVec.push_back(dist);     //Storing the distance to each image of currClass
			}

		}

		sort(distVec.begin(), distVec.end());   // Sort the distances to each image of currClass

		// sum distances to nearest K elements
		// check for k images or not
		if (distVec.size() >= k) {
			double sumD = 0;
			for (int t = 0; t < k; t++) {
				sumD += distVec[t];
			}
		}
	}

	sort(matcher.begin(), matcher.end());
	//return matcher[0].second;


	std::vector<std::vector<float>> data2;
	std::vector<char*> classes2;

	std::vector<std::pair<float, std::string>> matcher2;

	for (int i = 0; i < data.size(); i++) {
		if (classes[0] == matcher[0].second)
		{
			data2.push_back(data[i]);
			classes2.push_back(classes[i]);
		}
	}

	
	//Computing standard deviations
	for (int i = 0; i < data.size(); i++) {
		float sum = 0, mean = 0, var = 0;
		//Mean
		mean = std::accumulate(data[i].begin(), data[i].end(), 0.0) / data[i].size();
		//Variance
		for (const auto& val : data[i]) {
			var += (val - mean) * (val - mean);
		}
		var /= data[i].size();
		stdDevs.push_back(sqrt(var));
	}


	for (int i = 0; i < data.size(); i++) {

		float sum = 0;
		for (int j = 0; j < data[i].size(); j++) {
			sum += abs(ft_vec[j] - data[i][j]);
		}
		float dist = sqrt(sum) / stdDevs[i];
		matcher2.push_back(std::make_pair(dist, classes[i]));

	}

	sort(matcher2.begin(), matcher2.end()); // sort on the basis of distances
	return matcher2[0].second; //return name of closest match

}
