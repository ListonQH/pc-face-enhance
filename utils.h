#pragma once

#include <direct.h>
#include <io.h>
#include <filesystem>
#include <string>
#include <vector>

namespace LqhUtil
{
	std::string GetParentPath(std::string file_path);

	bool IsFileExist(const std::string& name);

	bool IsDirExist(const std::string& file_path, bool is_create);

	std::vector<std::string> GetFilesPath(std::string path, std::string file_type);
}
