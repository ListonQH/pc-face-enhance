#pragma once

#include <iostream>
#include "NvInfer.h"

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};
