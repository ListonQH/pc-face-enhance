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

class GfpGanClassCudaGraphs : public GfpGanClass
{
public:
	GfpGanClassCudaGraphs(std::string name);
	~GfpGanClassCudaGraphs();

	cv::Mat	Infer(cv::Mat in_img);
	cv::Mat	InferCudaGraphs(cv::Mat in_img);
	bool	Init();
	
private:

	bool pPrepareCudaGraphs();
	bool pWarmUp();

	TRTLogger						p_trt_logger;
	nvinfer1::IRuntime*				p_trt_runtime;
	nvinfer1::ICudaEngine*			p_trt_engine;
	nvinfer1::IExecutionContext*	p_trt_infer_context;

	cudaStream_t					p_trt_cuda_stream;
	cudaGraph_t						p_trt_cuda_graph;
	cudaGraphExec_t					p_trt_cuda_instance;

	size_t	p_gpu_buffer_input_index;
	size_t	p_gpu_buffer_output_index;
	void*	p_gpu_buffer_infer[MODEL_IO_NUM];

	size_t	p_h_2_d_img_data_size;
	size_t	p_d_2_h_img_data_size;
};
