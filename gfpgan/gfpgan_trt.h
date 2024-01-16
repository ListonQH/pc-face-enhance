#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "assert.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "trt_logger.h"
#include "run_constant.h"

class GfpGanClassTRT
{
public:
	GfpGanClassTRT(std::string name);
	~GfpGanClassTRT();	

	cv::Mat	Infer(cv::Mat in_img);
	bool	Init();
	void	DisplayTestInfo();

private:
	bool	pLoadEngineFileToMemery();
	void	pPreProcess(cv::Mat input_img);
	void	pPostProcess();

	std::string p_test_name;
	double		p_test_infer_time_ms;
	size_t		p_test_infer_counter;

	// engine file length
	size_t	p_engine_file_length;
	char*	p_engine_file_buffer;

	cv::Mat p_infer_result;

	cv::Mat p_b_channel;
	cv::Mat p_g_channel;
	cv::Mat p_r_channel;
	std::vector<cv::Mat> p_split_result_vec;

	cv::Mat p_b_out_channel;
	cv::Mat p_g_out_channel;
	cv::Mat p_r_out_channel;
	std::vector<cv::Mat> p_merge_result_vec;

	TRTLogger						p_trt_logger;
	nvinfer1::IRuntime*				p_trt_runtime;
	nvinfer1::ICudaEngine*			p_trt_engine;
	nvinfer1::IExecutionContext*	p_trt_infer_context;
	cudaStream_t					p_cuda_stream;

	size_t p_gpu_buffer_input_index;
	size_t p_gpu_buffer_output_index;
	void* p_gpu_buffer_infer[MODEL_IO_NUM];
	INPUT_DATA_TYPE*	p_cpu_infer_input_buffer;
	OUTPUT_DATA_TYPE*	p_cpu_infer_output_buffer;
};

