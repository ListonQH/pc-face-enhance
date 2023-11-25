#include "gfpgan_class_v85.h"

GfpGanClassV85::GfpGanClassV85(std::string name)
{	
	this->p_test_name = name;
	this->p_trt_runtime = nullptr;
	this->p_trt_engine = nullptr;
	this->p_trt_infer_context = nullptr;

	this->p_gpu_buffer_input_index = -1;
	this->p_gpu_buffer_output_index = -1;
}

GfpGanClassV85::~GfpGanClassV85()
{
	cudaStreamDestroy(p_trt_cuda_stream);
	CHECK(cudaFree(p_gpu_buffer_infer[0]));
	CHECK(cudaFree(p_gpu_buffer_infer[1]));

	p_trt_infer_context->destroy();
	p_trt_engine->destroy();
	p_trt_runtime->destroy();
}

cv::Mat GfpGanClassV85::Infer(cv::Mat in_img)
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
	size_t img_data_size = sizeof(INPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
	CHECK(cudaMemcpyAsync(p_gpu_buffer_infer[p_gpu_buffer_input_index],
		this->p_cpu_infer_input_buffer,
		img_data_size,
		cudaMemcpyHostToDevice, this->p_trt_cuda_stream));
	double begin = cv::getTickCount();
	// inference
	this->p_trt_infer_context->enqueueV2(p_gpu_buffer_infer, this->p_trt_cuda_stream, nullptr);
	//this->p_trt_infer_context->enqueueV3(this->p_trt_cuda_stream);
	
	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;

	// pull cuda to ram
	CHECK(cudaMemcpyAsync(this->p_cpu_infer_output_buffer,
		p_gpu_buffer_infer[p_gpu_buffer_output_index],
		img_data_size, cudaMemcpyDeviceToHost, this->p_trt_cuda_stream));

	// sync
	cudaStreamSynchronize(this->p_trt_cuda_stream);

	// get output
	this->pPostProcess();
	return this->p_infer_result;
}

bool GfpGanClassV85::Init()
{

	// Prepare engine file
	if (!this->pLoadEngineFileToMemery())
	{
		std::cerr << "Stop Build: read engine file error ." << std::endl;
		return false;
	}

	this->p_trt_runtime = nvinfer1::createInferRuntime(p_trt_logger);
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

	CHECK(cudaStreamCreate(&this->p_trt_cuda_stream));

	// check loaded model io num
	assert(this->p_trt_engine->getNbBindings() == 2);

	this->p_gpu_buffer_input_index = this->p_trt_engine->getBindingIndex(MODEL_INPUT_NAME);
	this->p_gpu_buffer_output_index = this->p_trt_engine->getBindingIndex(MODEL_OUTPUT_NAME);

	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_input_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(INPUT_DATA_TYPE)));

	CHECK(cudaMalloc(&p_gpu_buffer_infer[p_gpu_buffer_output_index],
		INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(INPUT_DATA_TYPE)));

	std::cout << "Build Finish ....." << std::endl;

	return true;
}
