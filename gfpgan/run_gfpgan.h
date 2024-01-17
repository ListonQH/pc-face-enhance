#pragma once

#include "gfpgan_trt.h"
#include "gfpgan_libtorch.h"
#include "gfpgan_mulit_stream.h"
#include "../utils.h"

#include <iostream>

bool run_gfpgan_trt();

bool run_gfpgan_libtorch();

bool run_gfpgan_cuda_ocv();

bool run_gfpgan_multi_stream();