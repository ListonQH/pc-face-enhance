#pragma once

#include "NvInfer.h"

#include <direct.h>
#include <io.h>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>

namespace LqhUtil
{
	std::string GetParentPath(std::string file_path);

	bool IsFileExist(const std::string& name);

	bool IsDirExist(const std::string& file_path, bool is_create);

	std::vector<std::string> GetFilesPath(std::string path, std::string file_type);


	std::string GetLayerDataType(nvinfer1::DataType dt);
	std::string GetLayerOpType(nvinfer1::LayerType lt);

	void print(std::string str);
	void print_err(std::string str);
}
