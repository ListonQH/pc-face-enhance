#include <iostream>
#include <string>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>

#include "error_code.h"
#include "gfpgan_class_v85.h"
#include "gfpgan_class_v86.h"

#define V86 true

using namespace std;

int main222()
{
	GfpGanClass* instance = nullptr;

#ifdef V85

	instance = new GfpGanClassV85("GfpGanClassV85");

#elif V86

	instance = new GfpGanClassV86("GfpGanClassV86");
	
#endif // V85 or V86

	assert(instance == nullptr);

	if (!instance->Init())
	{
		std::cerr << "Init gfpgan instance error. " << std::endl;
		return ERROR_MAIN_EXIT_GFPGAN_INIT;
	}
		
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Open Camera Faild ..." << endl;
		return -1;
	}
	cv::namedWindow("Old img");
	cv::namedWindow("New img");
	cv::Mat old_img;
	cv::Mat source_img;
	cv::Mat resize_img;
	cv::Mat new_img;
	int key = 0;
	size_t counter = 0;
	size_t out_infer_time = 0;
	size_t begin = 0;
	while (true)
	{
		cap >> old_img;
		cv::flip(old_img, source_img, 1);
		counter++;
		cv::resize(source_img, resize_img, cv::Size(512, 512));
		cv::imshow("Old img", resize_img);

		begin = cv::getTickCount();
		new_img = instance->Infer(resize_img);
		out_infer_time = out_infer_time + (cv::getTickCount() - begin);

		cv::imshow("New img", new_img);
		if (counter % 90 == 0)
		{
			instance->DisplayTestInfo();
			std::cout << instance->GetInstanceName() << "out infer time:" << out_infer_time * 1.0 / cv::getTickFrequency() * 1000.0  / counter << " ms" << std::endl << std::endl;
			counter = 0;
			out_infer_time = 0;
		}

		key = cv::waitKey(2);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	cv::destroyAllWindows();
	cap.release();
	return 0;
}