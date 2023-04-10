// imgDisplay.cpp
//This program reads an address and displays the image at that address.
// At keypress 'q', the program terminates.

#include <opencv2/opencv.hpp>
#include <iostream>


int main(int argc, char** argv)
{
	// Read the image file
	cv::Mat image = cv::imread("project_img.jpg");

	// Check for failure
	if (image.empty())
	{
		std::cout << "Image Not Found!!!" << std::endl;
		std::cin.get(); //wait for any key press
		return -1;
	}

	// Show the image inside a window.

	cv::imshow("Image Output", image);

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
