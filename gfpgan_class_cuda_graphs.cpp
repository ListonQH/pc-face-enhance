#include "gfpgan_class_cuda_graphs.h"

GfpGanClassCudaGraphs::GfpGanClassCudaGraphs(std::string name)
{
	p_test_name = "[" + name + "] ";
	p_trt_runtime = nullptr;
	p_trt_engine = nullptr;
	p_trt_infer_context = nullptr;

	p_gpu_buffer_input_index = 0;
	p_gpu_buffer_output_index = 0;
}

GfpGanClassCudaGraphs::~GfpGanClassCudaGraphs()
{
}

cv::Mat GfpGanClassCudaGraphs::Infer(cv::Mat in_img)
{
	if (in_img.channels() != INPUT_CHANNEL
		|| in_img.rows != INPUT_HEIGHT
		|| in_img.cols != INPUT_WIDTH)
	{
		std::cerr << "Input Face Image Shape error. Input (c * h * w): " << in_img.channels()
			<< "*" << in_img.rows << " * " << in_img.cols << ". Target (c * h * w): "
			<< INPUT_CHANNEL << "*" << INPUT_HEIGHT << "*" << INPUT_WIDTH << std::endl;
		return cv::Mat();
	}
	double begin = cv::getTickCount();
	this->pPreProcess(in_img);

	CHECK(cudaMemcpy(p_gpu_buffer_infer[p_gpu_buffer_input_index],
		this->p_cpu_infer_input_buffer,
		p_h_2_d_img_data_size,
		cudaMemcpyHostToDevice));

	//this->p_trt_infer_context->enqueueV3(this->p_trt_cuda_stream);
	cudaGraphLaunch(p_cuda_instance, p_cuda_streams[0]);
	cudaStreamSynchronize(p_cuda_streams[0]);

	CHECK(cudaMemcpy(this->p_cpu_infer_output_buffer,
		p_gpu_buffer_infer[p_gpu_buffer_output_index],
		p_d_2_h_img_data_size, cudaMemcpyDeviceToHost));

	// get output
	this->pPostProcess();
	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;
	return this->p_infer_result;
}

bool GfpGanClassCudaGraphs::Init()
{
	std::cout << "[GfpGanClassCudaGraphs] Init() Begin ..." << std::endl;

	p_h_2_d_img_data_size = sizeof(INPUT_DATA_TYPE)	 * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
	p_d_2_h_img_data_size = sizeof(OUTPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;

	// Prepare engine file
	if (!this->pLoadEngineFileToMemery())
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: read engine file error ." << std::endl;
		return false;
	}

	p_trt_runtime = nvinfer1::createInferRuntime(p_trt_logger);
	if (this->p_trt_runtime == nullptr)
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: create trt_runtime error ." << std::endl;
		return false;
	}

	this->p_trt_engine = this->p_trt_runtime->deserializeCudaEngine(this->p_engine_file_buffer, this->p_engine_file_length);
	// must to release memeary
	delete[] this->p_engine_file_buffer;

	if (this->p_trt_engine == nullptr)
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: deserialize cuda engine from file stream error ." << std::endl;
		return false;
	}

	auto stream_nb = p_trt_engine->getNbAuxStreams();
	std::cout << "[GfpGanClassCudaGraphs] Engine opened stream: " << stream_nb << std::endl;

	this->p_trt_infer_context = this->p_trt_engine->createExecutionContext();

	if (this->p_trt_infer_context == nullptr)
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: create inference context error ." << std::endl;
		return false;
	}

	// kHALF
	this->p_gpu_buffer_input_index = this->p_trt_engine->getBindingIndex(MODEL_INPUT_NAME);
	// f16: kFLOAT
	this->p_gpu_buffer_output_index = this->p_trt_engine->getBindingIndex(MODEL_OUTPUT_NAME);

	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_input_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(INPUT_DATA_TYPE)));

	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_output_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(OUTPUT_DATA_TYPE)));

	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html
	auto ret = this->p_trt_infer_context->setTensorAddress(MODEL_INPUT_NAME, p_gpu_buffer_infer[p_gpu_buffer_input_index]);
	if (!ret)
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: bind context output-tensor error ." << std::endl;
		return false;
	}

	ret = this->p_trt_infer_context->setTensorAddress(MODEL_OUTPUT_NAME, p_gpu_buffer_infer[p_gpu_buffer_output_index]);
	if (!ret)
	{
		std::cerr << "[GfpGanClassCudaGraphs] Stop Build: bind context output-tensor error ." << std::endl;
		return false;
	}
	std::cout << "[GfpGanClassCudaGraphs] Init() Finish and Passed! ....." << std::endl;

#ifdef USE_CUDA_GRAPHS

	std::cout << "[GfpGanClassCudaGraphs] USE_CUDA_GRAPHS Open and Begin pPrepareCudaGraphs() ..." << std::endl;
	ret = pPrepareCudaGraphs();
	if (!ret)
	{
		return false;
	}
#endif // USE_CUDA_GRAPHS

	return true;
}

bool GfpGanClassCudaGraphs::pPrepareCudaGraphs()
{
	std::cout << "[GfpGanClassCudaGraphs] pPrepareCudaGraphs() begin ..." << std::endl;

	auto ret = cudaStreamCreate(&p_cuda_streams[0]);
	if (ret != cudaSuccess)
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->cudaStreamCreate() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}
	if (!p_trt_infer_context->enqueueV3(p_cuda_streams[0]))
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->enqueueV3() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}

	ret = cudaStreamBeginCapture(p_cuda_streams[0], cudaStreamCaptureModeGlobal);
	if (ret != cudaSuccess)
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->cudaStreamBeginCapture() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}

	if (!p_trt_infer_context->enqueueV3(p_cuda_streams[0]))
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->enqueueV3() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}

	ret = cudaStreamEndCapture(p_cuda_streams[0], &p_cuda_graph);
	if (ret != cudaSuccess)
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->cudaStreamEndCapture() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}
	ret = cudaGraphInstantiate(&p_cuda_instance, p_cuda_graph, 0);
	if (ret != cudaSuccess)
	{
		std::cerr << "[GfpGanClassCudaGraphs] " << "pPrepareCudaGraphs()->cudaGraphInstantiate() Faild: " << cudaGetErrorString(ret)
			<< std::endl;
		return false;
	}

	cudaGraphDestroy(p_cuda_graph);
	
	std::cout << "[GfpGanClassCudaGraphs] pPrepareCudaGraphs() Finish and Passed ..." << std::endl;

	return true;
}
