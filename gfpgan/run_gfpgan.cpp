#include "run_gfpgan.h"

bool run_gfpgan_trt()
{
	GfpGanClassTRT* instance = new GfpGanClassTRT("GfpGanClassTRT");

	assert(instance == nullptr);

	if (!instance->Init())
	{
		LqhUtil::print_err("Init gfpgan instance error ... ");
		return false;
	}

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		LqhUtil::print_err("Open Camera Faild ...");
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
			std::cout << "[Main Thread] Out-Infer Time:" << out_infer_time * 1.0 / cv::getTickFrequency() * 1000.0 / counter << " ms" << std::endl << std::endl;
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

bool run_gfpgan_libtorch()
{
	return false;
}

bool run_gfpgan_cuda_ocv()
{
	return false;

}

bool run_gfpgan_multi_stream()
{	
	GfpGanMulitStream* instance = new GfpGanMulitStream("GfpGanMulitStream");

	assert(instance == nullptr);

	if (!instance->Init())
	{
		LqhUtil::print_err("Init gfpgan instance error ... ");
		return false;
	}

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		LqhUtil::print_err("Open Camera Faild ...");
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
			std::cout << "[Main Thread] Out-Infer Time:" << out_infer_time * 1.0 / cv::getTickFrequency() * 1000.0 / counter << " ms" << std::endl << std::endl;
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
