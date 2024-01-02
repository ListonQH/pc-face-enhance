#include "gfpgan_class_v86.h"

GfpGanClassV86::GfpGanClassV86(std::string name)
{
	p_test_name = "[" + name + "] ";
	p_trt_runtime = nullptr;
	p_trt_engine = nullptr;
	p_trt_infer_context = nullptr;

	p_gpu_buffer_input_index = 0;
	p_gpu_buffer_output_index = 0;
}

GfpGanClassV86::~GfpGanClassV86()
{

}

cv::Mat GfpGanClassV86::Infer(cv::Mat in_img)
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

	this->pPreProcess(in_img);

	// push ram to cuda
	size_t h_2_d_img_data_size = sizeof(INPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
	size_t d_2_h_img_data_size = sizeof(INPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
	
	double begin = cv::getTickCount();
	
	// inference
#ifdef ASYNC
	CHECK(cudaMemcpyAsync(p_gpu_buffer_infer[p_gpu_buffer_input_index],
		this->p_cpu_infer_input_buffer,
		h_2_d_img_data_size,
		cudaMemcpyHostToDevice, this->p_trt_cuda_stream));

	this->p_trt_infer_context->enqueueV3(this->p_trt_cuda_stream);
#else
	CHECK(cudaMemcpy(p_gpu_buffer_infer[p_gpu_buffer_input_index],
		this->p_cpu_infer_input_buffer,
		h_2_d_img_data_size,
		cudaMemcpyHostToDevice));
	this->p_trt_infer_context->executeV2(p_gpu_buffer_infer);
#endif // ASYNC

	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;

	// pull cuda to RAM
#ifdef ASYNC
	CHECK(cudaMemcpyAsync(this->p_cpu_infer_output_buffer,
		p_gpu_buffer_infer[p_gpu_buffer_output_index],
		d_2_h_img_data_size, cudaMemcpyDeviceToHost, this->p_trt_cuda_stream));
#else
	CHECK(cudaMemcpy(this->p_cpu_infer_output_buffer,
		p_gpu_buffer_infer[p_gpu_buffer_output_index],
		d_2_h_img_data_size, cudaMemcpyDeviceToHost));
#endif // ASYNC
	
	// sync
	//cudaStreamSynchronize(this->p_trt_cuda_stream);
	// get output
	this->pPostProcess();
	return this->p_infer_result;
}

bool GfpGanClassV86::Init()
{
	// Prepare engine file
	if (!this->pLoadEngineFileToMemery())
	{
		std::cerr << "Stop Build: read engine file error ." << std::endl;
		return false;
	}

	p_trt_runtime = nvinfer1::createInferRuntime(p_trt_logger);
	if (this->p_trt_runtime == nullptr)
	{
		std::cerr << "Stop Build: create trt_runtime error ." << std::endl;
		return false;
	}

	this->p_trt_engine = this->p_trt_runtime->deserializeCudaEngine(this->p_engine_file_buffer, this->p_engine_file_length);
	// must to release memeary
	delete[] this->p_engine_file_buffer;

	if (this->p_trt_engine == nullptr)
	{
		std::cerr << "Stop Build: deserialize cuda engine from file stream error ." << std::endl;
		return false;
	}

	this->p_trt_infer_context = this->p_trt_engine->createExecutionContext();
	if (this->p_trt_infer_context == nullptr)
	{
		std::cerr << "Stop Build: create inference context error ." << std::endl;
		return false;
	}

	// kHALF
	this->p_gpu_buffer_input_index = this->p_trt_engine->getBindingIndex(MODEL_INPUT_NAME);
	// f16: kFLOAT
	this->p_gpu_buffer_output_index = this->p_trt_engine->getBindingIndex(MODEL_OUTPUT_NAME);

	{
		/*auto dt = this->p_trt_engine->getTensorDataType(MODEL_OUTPUT_NAME);
		switch (dt)
		{
		case nvinfer1::DataType::kFLOAT:
			std::cout << "kFLOAT" << std::endl;
			break;
		case nvinfer1::DataType::kHALF:
			std::cout << "kHALF" << std::endl;
			break;
		case nvinfer1::DataType::kINT8:
			break;
		case nvinfer1::DataType::kINT32:
			break;
		case nvinfer1::DataType::kBOOL:
			break;
		case nvinfer1::DataType::kUINT8:
			break;
		case nvinfer1::DataType::kFP8:
			break;
		default:
			break;
		}
		exit(0);*/
	}
	
	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_input_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(INPUT_DATA_TYPE)));

	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_output_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(OUTPUT_DATA_TYPE)));

	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html
	auto ret = this->p_trt_infer_context->setTensorAddress(MODEL_INPUT_NAME, p_gpu_buffer_infer[p_gpu_buffer_input_index]);
	if (!ret)
	{
		std::cerr << "Stop Build: bind context output-tensor error ." << std::endl;
		return false;
	}

	ret = this->p_trt_infer_context->setTensorAddress(MODEL_OUTPUT_NAME, p_gpu_buffer_infer[p_gpu_buffer_output_index]);
	if (!ret)
	{
		std::cerr << "Stop Build: bind context output-tensor error ." << std::endl;
		return false;
	}
	std::cout << "Build Finish ....." << std::endl;
    return true;
}

