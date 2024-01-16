#pragma once

#include <iostream>
#include "NvInfer.h"

class TRTLogger : public nvinfer1::ILogger
{
private:
    Severity severity_level;
public:
    TRTLogger()
    {
        severity_level = Severity::kVERBOSE;
    }

    TRTLogger(Severity severity_level)
    {
        this->severity_level = severity_level;
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= severity_level)
            std::cout << msg << std::endl;
    }
};
