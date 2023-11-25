#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "run_constant.h"

class GfpGanClass
{
public:
	GfpGanClass();
	void DisplayTestInfo();
	std::string GetInstanceName();
	virtual bool Init();
	virtual cv::Mat Infer(cv::Mat in_img);

protected:
	bool	pLoadEngineFileToMemery();
	void	pPreProcess(cv::Mat input_img);
	void	pPostProcess();

	std::string p_test_name;
	double p_test_infer_time_ms;
	size_t p_test_infer_counter;

	// engine file length
	size_t p_engine_file_length;
	char* p_engine_file_buffer;

	cv::Mat p_b_channel;
	cv::Mat p_g_channel;
	cv::Mat p_r_channel;
	std::vector<cv::Mat> p_split_result_vec;
	cv::Mat p_infer_result;

	INPUT_DATA_TYPE* p_cpu_infer_input_buffer;
	INPUT_DATA_TYPE* p_cpu_infer_output_buffer;
};

