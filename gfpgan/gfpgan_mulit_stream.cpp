#include "gfpgan_mulit_stream.h"

GfpGanMulitStream::GfpGanMulitStream(std::string name)
{
	this->p_test_name = "[" + name + "] ";
	p_h_2_d_img_data_size = INPUT_CHANNEL * INPUT_WIDTH * INPUT_HEIGHT;
	p_d_2_h_img_data_size = INPUT_CHANNEL * INPUT_WIDTH * INPUT_HEIGHT;

	p_cpu_infer_input_buffer  = new INPUT_DATA_TYPE[p_h_2_d_img_data_size];
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

	for (int i = 0; i < STREAM_NB; i++)
	{
		cudaStreamCreate(&p_cuda_streams[i]);
	}

	p_gpu_buffer_input_index = 0;
	p_gpu_buffer_output_index = 0;
	p_cuda_streams_index = 0;
}

GfpGanMulitStream::~GfpGanMulitStream()
{

}

cv::Mat GfpGanMulitStream::Infer(cv::Mat in_img)
{
	//p_merge_result_vec.clear();
	cv::split(in_img, this->p_split_result_vec);

	double begin = cv::getTickCount();

	CHECK(cudaMemcpyAsync(p_gpu_buffer_infer[p_gpu_buffer_input_index],
		this->p_cpu_infer_input_buffer,
		p_h_2_d_img_data_size,
		cudaMemcpyHostToDevice, 
		p_cuda_streams[p_cuda_streams_index]));

	this->p_trt_infer_context->enqueueV3(p_cuda_streams[p_cuda_streams_index]);

	CHECK(cudaMemcpyAsync(this->p_cpu_infer_output_buffer,
		p_gpu_buffer_infer[p_gpu_buffer_output_index],
		p_d_2_h_img_data_size, 
		cudaMemcpyDeviceToHost,
		p_cuda_streams[p_cuda_streams_index]));
		
	cudaStreamSynchronize(this->p_cuda_streams[p_cuda_streams_index]);
			
	cv::merge(this->p_merge_result_vec, this->p_infer_result);

	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;
	p_cuda_streams_index = (p_cuda_streams_index + 1) % STREAM_NB;
	return this->p_infer_result;
}

bool GfpGanMulitStream::Init()
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

void GfpGanMulitStream::DisplayTestInfo()
{
	LqhUtil::print(p_test_name + "Frames:\t" + std::to_string(this->p_test_infer_counter) + "ms");
	LqhUtil::print(p_test_name + "Average: " + std::to_string(this->p_test_infer_time_ms / this->p_test_infer_counter));

	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0;
}

bool GfpGanMulitStream::pLoadEngineFileToMemery()
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

	std::cout << this->p_test_name << "Finish load plan(engine) file:" << std::endl;
	std::cout << this->p_test_name << "Engine file length:\t" << this->p_engine_file_length << std::endl;
	std::cout << this->p_test_name << "Engine file name:\t" << TRT_ENGINE_FILE_PATH << std::endl;

	return true;
}
