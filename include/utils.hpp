#pragma once

#ifndef UTILIS_HPP
#define UTILIS_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

namespace utils{

    int64_t get_num_parameters(torch::nn::Module &model);
    // method to visualize output prediction
    // method to calculate training time
    // method to calculate RECALL, precsion and f1 score 
    // load model in C++

}



#endif // UTILIS_HPP