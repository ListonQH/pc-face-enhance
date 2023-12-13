#include "utils.h"

std::string LqhUtil::GetParentPath(std::string filePath)
{
    size_t pos = filePath.find_last_of('\\');
    if (std::string::npos != pos && 0 != pos) {
        return filePath.substr(0, pos);
    }
    return "\\";
}

bool LqhUtil::IsFileExist(const std::string& name)
{
    FILE* fp = NULL;
    errno_t err = fopen_s(&fp, name.c_str(), "r");
    if (err == 0 && fp != NULL) {
        fclose(fp);
        return true;
    }
    else {
        return false;
    }
}

bool LqhUtil::IsDirExist(const std::string& filePath, bool isCreate)
{
    if (_access(filePath.c_str(), 0) == -1) {
        if (isCreate) {
            std::string parentPath = GetParentPath(filePath);
            if (parentPath.empty() || (0 == parentPath.compare("\\"))) {
                return false;
            }
            
            if (!IsDirExist(parentPath, true)) {
                return false;
            }

            _mkdir(filePath.c_str());

            return true;
        }
        else {
            return false;
        }
    }
    return true;
}

std::vector<std::string> LqhUtil::GetFilesPath(std::string path, std::string file_type)
{
    std::vector<std::string> ret;
    if (!IsDirExist(path, false))
    {
        // no necessary
        ret.clear();
        return ret;
    }

    for (auto& i : std::filesystem::directory_iterator(path)) {
        std::string t = i.path().string();
        if (t.ends_with(file_type))
        {
            ret.push_back(t);
        }
    }

    return ret;
}
