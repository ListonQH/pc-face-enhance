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

std::string LqhUtil::GetLayerDataType(nvinfer1::DataType dt)
{
    std::string ret = "Unknown";
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT:
        ret = "kFLOAT";
        break;
    case nvinfer1::DataType::kHALF:
        ret = "kHALF";
        break;
    case nvinfer1::DataType::kINT8:
        ret = "kINT8";
        break;
    case nvinfer1::DataType::kINT32:
        ret = "kINT32";
        break;
    case nvinfer1::DataType::kBOOL:
        ret = "kBOOL";
        break;
    case nvinfer1::DataType::kUINT8:
        ret = "kUINT8";
        break;
    case nvinfer1::DataType::kFP8:
        ret = "kFP8";
        break;
    default:
        break;
    }
    return ret;
}

std::string LqhUtil::GetLayerOpType(nvinfer1::LayerType lt)
{
    std::string ret = "Unknown";
    switch (lt)
    {
    case nvinfer1::LayerType::kCONVOLUTION:
        ret = "kCONVOLUTION";
        break;
    case nvinfer1::LayerType::kFULLY_CONNECTED:
        ret = "kFULLY_CONNECTED";
        break;
    case nvinfer1::LayerType::kACTIVATION:
        ret = "kACTIVATION";
        break;
    case nvinfer1::LayerType::kPOOLING:
        ret = "kPOOLING";
        break;
    case nvinfer1::LayerType::kLRN:
        ret = "kLRN";
        break;
    case nvinfer1::LayerType::kSCALE:
        ret = "kSCALE";
        break;
    case nvinfer1::LayerType::kSOFTMAX:
        ret = "kSOFTMAX";
        break;
    case nvinfer1::LayerType::kDECONVOLUTION:
        ret = "kDECONVOLUTION";
        break;
    case nvinfer1::LayerType::kCONCATENATION:
        ret = "kCONCATENATION";
        break;
    case nvinfer1::LayerType::kELEMENTWISE:
        ret = "kELEMENTWISE";
        break;
    case nvinfer1::LayerType::kPLUGIN:
        ret = "kPLUGIN";
        break;
    case nvinfer1::LayerType::kUNARY:
        ret = "kUNARY";
        break;
    case nvinfer1::LayerType::kPADDING:
        ret = "kPADDING";
        break;
    case nvinfer1::LayerType::kSHUFFLE:
        ret = "kSHUFFLE";
        break;
    case nvinfer1::LayerType::kREDUCE:
        ret = "kREDUCE";
        break;
    case nvinfer1::LayerType::kTOPK:
        ret = "kTOPK";
        break;
    case nvinfer1::LayerType::kGATHER:
        ret = "kGATHER";
        break;
    case nvinfer1::LayerType::kMATRIX_MULTIPLY:
        ret = "kMATRIX_MULTIPLY";
        break;
    case nvinfer1::LayerType::kRAGGED_SOFTMAX:
        ret = "kRAGGED_SOFTMAX";
        break;
    case nvinfer1::LayerType::kCONSTANT:
        ret = "kCONSTANT";
        break;
    case nvinfer1::LayerType::kRNN_V2:
        ret = "kRNN_V2";
        break;
    case nvinfer1::LayerType::kIDENTITY:
        ret = "kIDENTITY";
        break;
    case nvinfer1::LayerType::kPLUGIN_V2:
        ret = "kPLUGIN_V2";
        break;
    case nvinfer1::LayerType::kSLICE:
        ret = "kSLICE";
        break;
    case nvinfer1::LayerType::kSHAPE:
        ret = "kSHAPE";
        break;
    case nvinfer1::LayerType::kPARAMETRIC_RELU:
        ret = "kPARAMETRIC_RELU";
        break;
    case nvinfer1::LayerType::kRESIZE:
        ret = "kRESIZE";
        break;
    case nvinfer1::LayerType::kTRIP_LIMIT:
        ret = "kTRIP_LIMIT";
        break;
    case nvinfer1::LayerType::kRECURRENCE:
        ret = "kRECURRENCE";
        break;
    case nvinfer1::LayerType::kITERATOR:
        ret = "kITERATOR";
        break;
    case nvinfer1::LayerType::kLOOP_OUTPUT:
        ret = "kLOOP_OUTPUT";
        break;
    case nvinfer1::LayerType::kSELECT:
        ret = "kSELECT";
        break;
    case nvinfer1::LayerType::kFILL:
        ret = "kFILL";
        break;
    case nvinfer1::LayerType::kQUANTIZE:
        ret = "kQUANTIZE";
        break;
    case nvinfer1::LayerType::kDEQUANTIZE:
        ret = "kDEQUANTIZE";
        break;
    case nvinfer1::LayerType::kCONDITION:
        ret = "kCONDITION";
        break;
    case nvinfer1::LayerType::kCONDITIONAL_INPUT:
        ret = "kCONDITIONAL_INPUT";
        break;
    case nvinfer1::LayerType::kCONDITIONAL_OUTPUT:
        ret = "kCONDITIONAL_OUTPUT";
        break;
    case nvinfer1::LayerType::kSCATTER:
        ret = "kSCATTER";
        break;
    case nvinfer1::LayerType::kEINSUM:
        ret = "kEINSUM";
        break;
    case nvinfer1::LayerType::kASSERTION:
        ret = "kASSERTION";
        break;
    case nvinfer1::LayerType::kONE_HOT:
        ret = "kONE_HOT";
        break;
    case nvinfer1::LayerType::kNON_ZERO:
        ret = "kNON_ZERO";
        break;
    case nvinfer1::LayerType::kGRID_SAMPLE:
        ret = "kGRID_SAMPLE";
        break;
    case nvinfer1::LayerType::kNMS:
        ret = "kNMS";
        break;
    case nvinfer1::LayerType::kREVERSE_SEQUENCE:
        ret = "kREVERSE_SEQUENCE";
        break;
    case nvinfer1::LayerType::kNORMALIZATION:
        ret = "kNORMALIZATION";
        break;
    case nvinfer1::LayerType::kCAST:
        ret = "kCAST";
        break;
    default:
        break;
    }
    return ret;
}

void LqhUtil::print(std::string str)
{
    std::cout << str << std::endl;
}

void LqhUtil::print_err(std::string str)
{
    std::cerr << str << std::endl;
}
