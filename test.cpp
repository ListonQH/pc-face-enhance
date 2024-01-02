#include <iostream>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "gfpgan_class_v86.h"
#include "utils.h"
#include "NvOnnxParser.h"
#include "gfpgan_trt_int8_calibrator.h"

void test_case_export_engine_from_onnx_INT8()
{
	std::cout << "[TEST CASE] test_case_export_engine_feom_onnx begin ..." << std::endl;
	GfpGanClass obj;
	std::string onnx_path = "D:\\codes\\onnx_tool\\v12f16.onnx";
	std::string engine_path = "";

	if (!LqhUtil::IsFileExist(onnx_path))
	{
		std::cerr << "[GfpGanClass] GenerateEngineFromONNX stop! onnx_path: " << onnx_path << " not found! " << std::endl;
		return;
	}

	nvinfer1::BuilderFlag build_flag = nvinfer1::BuilderFlag::kINT8;
	std::string engine_save_path = engine_path + "v12i8.engine";

	TRTLogger logger(nvinfer1::ILogger::Severity::kINFO);

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kINFO));

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	if ((build_flag != nvinfer1::BuilderFlag::kFP16)
		&&
		(build_flag != nvinfer1::BuilderFlag::kTF32)
		&&
		(build_flag != nvinfer1::BuilderFlag::kINT8)
		)
	{
		std::cerr << "[GfpGanClass] GenerateEngineFromONNX only support [F16, F32, INT8] data type!" << std::endl;

		delete parser;
		delete network;
		delete config;
		delete builder;

		return;
	}

	config->setFlag(build_flag);

	if (nvinfer1::BuilderFlag::kINT8 == build_flag)
	{
		//std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
		//calibrator.reset(new SFInt8Calibrator());
		//nvinfer1::IInt8Calibrator* calibrator = new SFInt8Calibrator();		
		config->setInt8Calibrator(new GfpGanInt8Calibrator());

		// if not set calibrator: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#set-dynamic-range
		//tensor->setDynamicRange(min_float, max_float);
	}

	nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

	std::ofstream outEngine(engine_save_path.c_str(), std::ios::binary);
	if (!outEngine)
	{
		std::cerr << "could not open output file" << std::endl;
		return;
	}
	outEngine.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

	delete parser;
	delete network;
	delete config;
	delete builder;
	delete serializedModel;

	std::cout << "[GfpGanClass] GenerateEngineFromONNX finish! Exported engine file path: " << engine_save_path << "." << std::endl;

	std::cout << "[TEST CASE] test_case_export_engine_feom_onnx end ..." << std::endl;
}

void layers()
{
	std::cout << "[TEST CASE] test_case_export_engine_feom_onnx begin ..." << std::endl;
	std::string onnx_path = "./models/fold.onnx";

	std::string engine_path = "./models/";
	std::string engine_save_path = engine_path + "fold.engine";


	if (!LqhUtil::IsFileExist(onnx_path))
	{
		std::cerr << "[GfpGanClass] GenerateEngineFromONNX stop! onnx_path: " << onnx_path << " not found! " << std::endl;
		return;
	}

	TRTLogger logger(nvinfer1::ILogger::Severity::kINFO);

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kINFO));

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
	config->setFlag(nvinfer1::BuilderFlag::kINT8);
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	config->setInt8Calibrator(new GfpGanInt8Calibrator());

	size_t laye_nb = network->getNbLayers();

	{
		/**********************  No **********************************/
		//for (int i = 0; i < laye_nb; i++)
		//{
		//	auto layer = network->getLayer(i);
		//	std::string layer_name = layer->getName();
		//	// find
		//	if (layer_name.find("weight") == std::string::npos
		//		&&
		//		layer_name.find("Constant") == std::string::npos)
		//	{
		//		if (layer_name.find("modulated_conv") != std::string::npos)
		//		{
		//			if (layer_name.find("to_rgb") == std::string::npos)
		//			{
		//				layer->setPrecision(nvinfer1::DataType::kHALF);
		//				std::cout << i << "\t" << layer_name << std::endl;
		//			}
		//		}
		//	}
		//}
	}

	for (int i = 0; i < laye_nb; i++)
	{
		auto layer = network->getLayer(i);
		auto lt = layer->getType();
		std::string layer_name = layer->getName();

		if (lt == nvinfer1::LayerType::kACTIVATION)
		{
			if (true)
			{
				layer->setPrecision(nvinfer1::DataType::kHALF);
			}

			if (false)
			{
				if (layer_name.find("/stylegan_decoder/style_conv") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if (layer_name.find("/conv_body_up.") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if (layer_name.find("/conv_body_down.") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if ((layer_name.find("/conv_body_down.") != std::string::npos)
					||
					(layer_name.find("/conv_body_up.") != std::string::npos)
					)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if ((layer_name.find("/condition_scale.") != std::string::npos))
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if ((layer_name.find("/condition_shift.") != std::string::npos))
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}


		}
		// lt == nvinfer1::LayerType::kCONVOLUTION 
		if (false)
		{

			// False
			if (false)
			{
				if (layer_name.find("/stylegan_decoder/style_conv") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if (layer_name.find("/stylegan_decoder/style_conv.") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if (layer_name.find("/stylegan_decoder/to_rgbs.") != std::string::npos)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}

			if (false)
			{
				if (layer_name.find("/stylegan_decoder/to_rgbs.") != std::string::npos
					||
					layer_name.find("/stylegan_decoder/style_conv.") != std::string::npos
					)
				{
					layer->setPrecision(nvinfer1::DataType::kHALF);
				}
			}
			if (true)
			{
				layer->setPrecision(nvinfer1::DataType::kHALF);
			}


		}
	}

	//return;
	nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

	std::ofstream outEngine(engine_save_path.c_str(), std::ios::binary);
	if (!outEngine)
	{
		std::cerr << "could not open output file" << std::endl;
		return;
	}
	outEngine.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

	delete parser;
	delete network;
	delete config;
	delete builder;
	delete serializedModel;

	std::cout << "[GfpGanClass] GenerateEngineFromONNX finish! Exported engine file path: " << engine_save_path << "." << std::endl;

	std::cout << "[TEST CASE] test_case_export_engine_feom_onnx end ..." << std::endl;
}

int mainss()
{
	if (cv::cuda::getCudaEnabledDeviceCount() == 0)
	{
		std::cerr << "Not Support Cuda" << std::endl;
	}
	else
	{
		std::cout << "Opencv Support Cuda" << std::endl;
	}

	//test_case_export_engine_from_onnx_INT8();

	layers();
}