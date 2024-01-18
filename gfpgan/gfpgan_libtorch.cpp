#include "gfpgan_libtorch.h"

GfpGanLibtorch::GfpGanLibtorch(std::string name)
{
	this->p_test_name = "[" + name + "] ";

	p_h_2_d_img_data_size = INPUT_CHANNEL * INPUT_WIDTH * INPUT_HEIGHT;
	p_d_2_h_img_data_size = INPUT_CHANNEL * INPUT_WIDTH * INPUT_HEIGHT;

	p_cpu_infer_input_buffer = new INPUT_DATA_TYPE[p_h_2_d_img_data_size];
	p_cpu_infer_output_buffer = new OUTPUT_DATA_TYPE[p_h_2_d_img_data_size];

	p_r_in_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_input_buffer + 0 * INPUT_WIDTH * INPUT_HEIGHT);
	p_g_in_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_input_buffer + 1 * INPUT_WIDTH * INPUT_HEIGHT);
	p_b_in_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_input_buffer + 2 * INPUT_WIDTH * INPUT_HEIGHT);
	p_split_result_vec.emplace_back(p_b_in_channel);
	p_split_result_vec.emplace_back(p_g_in_channel);
	p_split_result_vec.emplace_back(p_r_in_channel);

	p_r_out_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_output_buffer + 0 * INPUT_WIDTH * INPUT_HEIGHT);
	p_g_out_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_output_buffer + 1 * INPUT_WIDTH * INPUT_HEIGHT);
	p_b_out_channel = cv::Mat(512, 512, CV_8UC1, p_cpu_infer_output_buffer + 2 * INPUT_WIDTH * INPUT_HEIGHT);
	p_merge_result_vec.emplace_back(p_b_out_channel);
	p_merge_result_vec.emplace_back(p_g_out_channel);
	p_merge_result_vec.emplace_back(p_r_out_channel);
}

GfpGanLibtorch::~GfpGanLibtorch()
{
}

cv::Mat GfpGanLibtorch::Infer(cv::Mat in_img)
{
	double begin = cv::getTickCount();
	//p_merge_result_vec.clear();
	cv::split(in_img, this->p_split_result_vec);
	memcpy(p_cpu_infer_output_buffer, p_cpu_infer_input_buffer, 3 * 512 * 512);
	cv::merge(this->p_merge_result_vec, this->p_infer_result);

	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;	
	return this->p_infer_result;
}

bool GfpGanLibtorch::Init()
{
	LqhUtil::print(this->p_test_name + "torch::cuda::is_available: " + (torch::cuda::is_available()?"True":"False"));
	LqhUtil::print(this->p_test_name + "torch::cudnn::is_available: " + (torch::cuda::cudnn_is_available() ? "True" : "False"));
	return true;
}

void GfpGanLibtorch::DisplayTestInfo()
{
}
