#include <iostream>
#include <opencv2/opencv.hpp>

int mai11n()
{
	if (cv::cuda::getCudaEnabledDeviceCount() == 0)
	{
		std::cerr << "Not Support Cuda" << std::endl;
	}
	else
	{
		std::cout << "Opencv Support Cuda" << std::endl;
	}
}