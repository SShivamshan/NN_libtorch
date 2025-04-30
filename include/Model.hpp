#pragma once

#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "Architecture.hpp"
using namespace architecture;

namespace Model
{

    class ClassifierImpl: public torch::nn::Module{
        public:
            ClassifierImpl(const size_t image_size, std::vector<int> in_channels = {512, 256, 128, 64, 10},const int num_classes = 10);
            torch::Tensor forward(torch::Tensor x);
        private:
            torch::nn::Linear fc1{nullptr},fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
            torch::nn::Dropout dropout;
    };
    TORCH_MODULE(Classifier);


    class ConvNextClassifierImpl: public torch::nn::Module {
        public:
            ConvNextClassifierImpl(int num_classes = 100,
                    std::vector<int> in_channels = {64, 128, 256, 512, 1024},
                    const int image_channel = 3,
                    bool down_sample = true);

            torch::Tensor forward(torch::Tensor x);

        private:
            ConvNext _convnext{nullptr};
            torch::nn::AdaptiveAvgPool2d _avgpool{nullptr};
            torch::nn::Linear _classifier{nullptr};
            torch::nn::Dropout _dropout{nullptr};
    };

    TORCH_MODULE(ConvNextClassifier);
    

    
} // namespace Model



#endif // MODEL_HPP