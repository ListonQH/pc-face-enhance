#include "gfpgan_cuda_ocv.h"

GfpGanClassCudaOCV::GfpGanClassCudaOCV()
{
	p_test_name = "[GfpGanClassCudaOCV]";

	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;

	p_engine_file_length = -1;
	p_engine_file_buffer = nullptr;

	p_trt_runtime = nullptr;
	p_trt_engine = nullptr;
	p_trt_infer_context = nullptr;
}

void GfpGanClassCudaOCV::DisplayTestInfo()
{
	std::cout << p_test_name << p_test_infer_counter << " totall time (ms): " << p_test_infer_time_ms << " ms" << std::endl;
	std::cout << p_test_name << p_test_infer_counter << " average time (ms): " << p_test_infer_time_ms / p_test_infer_counter << " ms per frame" << std::endl;
	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;
}

std::string GfpGanClassCudaOCV::GetInstanceName()
{
	return p_test_name;
}

bool GfpGanClassCudaOCV::Init()
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

	this->p_gpu_buffer_input_index = this->p_trt_engine->getBindingIndex(MODEL_INPUT_NAME);
	this->p_gpu_buffer_output_index = this->p_trt_engine->getBindingIndex(MODEL_OUTPUT_NAME);

	return false;
}

cv::Mat GfpGanClassCudaOCV::Infer(cv::Mat in_img)
{
	return cv::Mat();
}

bool GfpGanClassCudaOCV::pLoadEngineFileToMemery()
{
	// binary read engine file
	std::ifstream engine_in_stream(TRT_ENGINE_FILE_PATH, std::ios::binary);
	if (!engine_in_stream)
	{
		std::cerr << "Error Load Engine File. File Path: " << TRT_ENGINE_FILE_PATH << std::endl;
		return false;
	}
	// get engne file length 
	engine_in_stream.seekg(0, engine_in_stream.end);
	this->p_engine_file_length = engine_in_stream.tellg();

	// set file-reader-pointer to head
	engine_in_stream.seekg(0, engine_in_stream.beg);

	// new buffer
	this->p_engine_file_buffer = new char[this->p_engine_file_length];

	// read file
	engine_in_stream.read(this->p_engine_file_buffer, this->p_engine_file_length);

	// close file stream
	engine_in_stream.close();

	return true;
}

void GfpGanClassCudaOCV::pPreProcess(cv::Mat input_img)
{
}

void GfpGanClassCudaOCV::pPostProcess()
{
}
