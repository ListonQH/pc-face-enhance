#include "gfpgan_trt.h"

GfpGanClassTRT::GfpGanClassTRT(std::string name)
{
	this->p_b_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);
	this->p_g_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);
	this->p_r_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);

	p_b_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);
	p_g_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);
	p_r_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);

	this->p_merge_result_vec.push_back(p_b_out_channel);
	this->p_merge_result_vec.push_back(p_g_out_channel);
	this->p_merge_result_vec.push_back(p_r_out_channel);

	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;

	p_engine_file_length = -1;
	p_engine_file_buffer = nullptr;

	p_test_name = "[" + name + "] ";
	p_trt_runtime = nullptr;
	p_trt_engine = nullptr;
	p_trt_infer_context = nullptr;

	p_gpu_buffer_input_index = 0;
	p_gpu_buffer_output_index = 0;

	p_cpu_infer_input_buffer = new INPUT_DATA_TYPE[INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH];
	p_cpu_infer_output_buffer = new OUTPUT_DATA_TYPE[INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH];
}

GfpGanClassTRT::~GfpGanClassTRT()
{

}

cv::Mat GfpGanClassTRT::Infer(cv::Mat in_img)
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

	// push ram to cuda
	size_t h_2_d_img_data_size = sizeof(INPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
	size_t d_2_h_img_data_size = sizeof(INPUT_DATA_TYPE) * INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
		
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

	p_test_infer_counter = p_test_infer_counter + 1;
	p_test_infer_time_ms = p_test_infer_time_ms + (cv::getTickCount() - begin) / cv::getTickFrequency() * 1000.0;

	return this->p_infer_result;
}

bool GfpGanClassTRT::Init()
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

bool GfpGanClassTRT::pLoadEngineFileToMemery()
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

void GfpGanClassTRT::pPreProcess(cv::Mat input_img)
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

void GfpGanClassTRT::pPostProcess()
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

void GfpGanClassTRT::DisplayTestInfo()
{
	std::cout << p_test_name << p_test_infer_counter << " average time (ms): " << p_test_infer_time_ms / p_test_infer_counter << " ms per frame" << std::endl;
	//std::cout << std::endl;
	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;
}