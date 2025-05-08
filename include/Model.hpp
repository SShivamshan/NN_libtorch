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
            ConvNextClassifierImpl(int num_classes, const int image_channel = 3, bool  use_tiny = false);
            torch::Tensor forward(torch::Tensor x);

        private:
            ConvNeXt convnext{nullptr};
            torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
            torch::nn::LayerNorm norm{nullptr};
            torch::nn::Linear fc{nullptr};
            torch::nn::Dropout dropout1{nullptr};
            torch::nn::Dropout dropout2{nullptr};
            torch::nn::Linear classifier{nullptr};
    };
    TORCH_MODULE(ConvNextClassifier);
    
    class MobileViTClassifierImpl: public torch::nn::Module{
        public:
            MobileViTClassifierImpl(int img_size, std::vector<int> features_list, std::vector<int> d_list,
                std::vector<int> transformer_depth, int expansion,int num_classes);
            
            torch::Tensor forward(torch::Tensor x);
        
        private:
            MobileViT _mobilevit{nullptr};
            torch::nn::AdaptiveAvgPool2d _avgpool{nullptr};
            torch::nn::Linear _fc{nullptr};
            torch::nn::BatchNorm1d _batchnorm{nullptr};  
            torch::nn::Linear _classifier{nullptr};
            torch::nn::Dropout _dropout{nullptr};
    };
    TORCH_MODULE(MobileViTClassifier);

} // namespace Model


#endif // MODEL_HPP