#include "gfpgan_class.h"

GfpGanClass::GfpGanClass()
{	
	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;

	p_engine_file_length = -1;
	p_engine_file_buffer = nullptr;

#ifdef MODEL_FP_16
	this->p_b_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_16FC1);
	this->p_g_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_16FC1);
	this->p_r_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_16FC1);

	p_b_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);
	p_g_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);
	p_r_out_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);

	this->p_merge_result_vec.push_back(this->p_b_out_channel);
	this->p_merge_result_vec.push_back(this->p_g_out_channel);
	this->p_merge_result_vec.push_back(this->p_r_out_channel);

#else
	this->p_b_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);
	this->p_g_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);
	this->p_r_channel = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1);
#endif // MODEL_FP_16

	this->p_split_result_vec.push_back(this->p_b_channel);
	this->p_split_result_vec.push_back(this->p_g_channel);
	this->p_split_result_vec.push_back(this->p_r_channel);
	

	p_cpu_infer_input_buffer = new INPUT_DATA_TYPE[INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH];
	p_cpu_infer_output_buffer = new OUTPUT_DATA_TYPE[INPUT_BATCH * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH];

}

void GfpGanClass::DisplayTestInfo()
{
	std::cout << p_test_name << p_test_infer_counter << " totall time (ms): " << p_test_infer_time_ms << " ms" << std::endl;
	std::cout << p_test_name << p_test_infer_counter << " average time (ms): " << p_test_infer_time_ms / p_test_infer_counter << " ms per frame" << std::endl;
	//std::cout << std::endl;
	p_test_infer_counter = 0;
	p_test_infer_time_ms = 0.0f;
}

std::string GfpGanClass::GetInstanceName()
{
	return p_test_name;
}

bool GfpGanClass::Init()
{
	return false;
}

cv::Mat GfpGanClass::Infer(cv::Mat in_img)
{
	return cv::Mat();
}

void GfpGanClass::pPreProcess(cv::Mat input_img)
{
	cv::Mat input_mat;
	// conver to float: 32f
#ifdef MODEL_FP_16
	input_img.convertTo(input_mat, CV_16FC3, 1.0 / 255.0);
#else
	input_img.convertTo(input_mat, CV_32FC3, 1.0 / 255.0);
#endif // MODEL_FP_16

	
	// in gfpgan: mean: 0.5; STD:0.5
	cv::Mat input_data = (input_mat - 0.5) / 0.5; // what about (2 * input_mat - 1)
	cv::split(input_data, this->p_split_result_vec);
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

void GfpGanClass::pPostProcess()
{
	size_t img_data_size = (INPUT_HEIGHT) * (INPUT_WIDTH);
	
#ifdef MODEL_FP_16
	memcpy(this->p_r_out_channel.data, this->p_cpu_infer_output_buffer + 0 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_g_out_channel.data, this->p_cpu_infer_output_buffer + 1 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_b_out_channel.data, this->p_cpu_infer_output_buffer + 2 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	cv::merge(this->p_merge_result_vec, this->p_infer_result);
#else
	memcpy(this->p_r_channel.data, this->p_cpu_infer_output_buffer + 0 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_g_channel.data, this->p_cpu_infer_output_buffer + 1 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	memcpy(this->p_b_channel.data, this->p_cpu_infer_output_buffer + 2 * img_data_size, img_data_size * sizeof(OUTPUT_DATA_TYPE));
	cv::merge(this->p_split_result_vec, this->p_infer_result);
#endif // MODEL_FP_16

}

bool GfpGanClass::pLoadEngineFileToMemery()
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
	std::cout << this->p_test_name << "Engine file length:\t" << this-> p_engine_file_length << std::endl;
	std::cout << this->p_test_name << "Engine file name:\t" << TRT_ENGINE_FILE_PATH << std::endl;

	return true;
}
