#include <iostream>
#include <string>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>

#include "error_code.h"
#include "gfpgan/run_gfpgan.h"

using namespace std;

enum GfpGanEnum
{
	TRT,
	Libtorch,
};

int main()
{
	auto run_gfpgan_type = GfpGanEnum::Libtorch;
	switch (run_gfpgan_type)
	{
	case TRT:
		run_gfpgan_trt();		
		break;
	case Libtorch:
		run_gfpgan_libtorch();
		break;
	default:
		break;
	}
	
}