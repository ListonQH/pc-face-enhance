#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "torch/torch.h"
#include "torch/script.h"
#include "run_constant.h"
#include "../utils.h"

class GfpGanLibtorch
{
public:
	GfpGanLibtorch(std::string name);
	~GfpGanLibtorch();

	cv::Mat	Infer(cv::Mat in_img);
	bool	Init();
	void	DisplayTestInfo();

private:
	std::string p_test_name;
	double		p_test_infer_time_ms;
	size_t		p_test_infer_counter;

	cv::Mat p_infer_result;

	cv::Mat p_b_in_channel;
	cv::Mat p_g_in_channel;
	cv::Mat p_r_in_channel;
	std::vector<cv::Mat> p_split_result_vec;

	cv::Mat p_b_out_channel;
	cv::Mat p_g_out_channel;
	cv::Mat p_r_out_channel;
	std::vector<cv::Mat> p_merge_result_vec;

	size_t	p_h_2_d_img_data_size;
	size_t	p_d_2_h_img_data_size;

	INPUT_DATA_TYPE*	p_cpu_infer_input_buffer;
	OUTPUT_DATA_TYPE*	p_cpu_infer_output_buffer;
};

