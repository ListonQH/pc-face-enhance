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
#include "gfpgan_class.h"

class GfpGanClassV85 : public GfpGanClass
{
public:
	GfpGanClassV85(std::string name);
	~GfpGanClassV85();
	cv::Mat Infer(cv::Mat in_img);
	bool Init();

private:

	TRTLogger						p_trt_logger;
	nvinfer1::IRuntime*				p_trt_runtime;
	nvinfer1::ICudaEngine*			p_trt_engine;
	nvinfer1::IExecutionContext*	p_trt_infer_context;
	cudaStream_t					p_trt_cuda_stream;

	size_t p_gpu_buffer_input_index;
	size_t p_gpu_buffer_output_index;
	void* p_gpu_buffer_infer[MODEL_IO_NUM];

};

