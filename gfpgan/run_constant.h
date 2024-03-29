#pragma once

#include <opencv2/opencv.hpp>

#define USE_CUDA_GRAPHS     true

#define MODEL_IO_UI8            true
#define INPUT_DATA_TYPE			unsigned char
#define OUTPUT_DATA_TYPE		unsigned char
constexpr auto TRT_ENGINE_FILE_PATH = "./models/folded.trt";

#define INPUT_BATCH				1
#define INPUT_CHANNEL			3
#define INPUT_WIDTH				512
#define INPUT_HEIGHT			512

constexpr auto MODEL_IO_NUM         = 2;
constexpr auto MODEL_INPUT_NAME     = "input";
constexpr auto MODEL_OUTPUT_NAME    = "output";

#define DATA_COPY_MEMORY  true
//#define ASYNC             true

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            std::cerr << "Assertion failure: " << #condition << std::endl;  \
            abort();                                                        \
        }                                                                   \
    } while (0)