#include "gfpgan_cuda_graphs.h"

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

bool GfpGanClassCudaGraphs::pLoadEngineFileToMemery()
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

void GfpGanClassCudaGraphs::pPreProcess(cv::Mat input_img)
{
	cv::Mat input_mat;
	// conver to float: 32f
#ifdef MODEL_IO_F16
	input_img.convertTo(input_mat, CV_16FC3, 1.0 / 255.0, 0);
#elif MODEL_IO_UI8
	// nothing 
#else
	input_img.convertTo(input_mat, CV_32FC3, 1.0 / 255.0);
#endif // MODEL_FP_16

#ifndef MODEL_IO_UI8
	// in gfpgan: mean: 0.5; STD:0.5
	cv::Mat input_data = (input_mat - 0.5f) / 0.5f; // what about (2 * input_mat - 1)
	cv::split(input_data, this->p_split_result_vec);
#else
	cv::split(input_img, this->p_split_result_vec);
#endif // !MODEL_IO_UI8


	double begin = cv::getTickCount();

	size_t img_data_size = (INPUT_HEIGHT) * (INPUT_WIDTH);

	// 0.05ms
	if (DATA_COPY_MEMORY)
	{
		memcpy(this->p_cpu_infer_input_buffer + 0 * img_data_size, this->p_split_result_vec[2].data, img_data_size * sizeof(INPUT_DATA_TYPE));
		memcpy(this->p_cpu_infer_input_buffer + 1 * img_data_size, this->p_split_result_vec[1].data, img_data_size * sizeof(INPUT_DATA_TYPE));
		memcpy(this->p_cpu_infer_input_buffer + 2 * img_data_size, this->p_split_result_vec[0].data, img_data_size * sizeof(INPUT_DATA_TYPE));
		//std::cout << "memcpy :" << (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
	}

	// 0.3ms
	if (!DATA_COPY_MEMORY)
	{
		for (int r = 0; r < INPUT_HEIGHT; r++)
		{
			INPUT_DATA_TYPE* rowData = input_img.ptr<INPUT_DATA_TYPE>(r);
			for (int c = 0; c < INPUT_WIDTH; c++)
			{
				p_cpu_infer_input_buffer[0 * img_data_size + r * INPUT_WIDTH + c] = rowData[INPUT_CHANNEL * c];
				p_cpu_infer_input_buffer[1 * img_data_size + r * INPUT_WIDTH + c] = rowData[INPUT_CHANNEL * c + 1];
				p_cpu_infer_input_buffer[2 * img_data_size + r * INPUT_WIDTH + c] = rowData[INPUT_CHANNEL * c + 2];
			}
		}
		std::cout << "for :" << (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
	}
}

void GfpGanClassCudaGraphs::pPostProcess()
{
	size_t img_data_size = (INPUT_HEIGHT) * (INPUT_WIDTH);

	memcpy(this->p_r_out_channel.data, this->p_cpu_infer_output_buffer + 0 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_g_out_channel.data, this->p_cpu_infer_output_buffer + 1 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_b_out_channel.data, this->p_cpu_infer_output_buffer + 2 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));

	cv::merge(this->p_merge_result_vec, this->p_infer_result);

#ifndef MODEL_IO_UI8
	this->p_infer_result = (this->p_infer_result + 1.0f) / 2.0f;
	this->p_infer_result.convertTo(this->p_infer_result, CV_8UC3, 255);
#endif // !MODEL_IO_UI8

}
