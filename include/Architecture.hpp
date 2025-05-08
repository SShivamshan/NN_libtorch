#pragma once

#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "Backbone.hpp"


using namespace backbone;

namespace architecture {

    class DownsampleImpl : public torch::nn::Module {
        public:
            torch::nn::LayerNorm norm{nullptr};
            torch::nn::Conv2d conv{nullptr};
            
            DownsampleImpl(int in_channels, int out_channels, int stride);
            
            torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(Downsample);
    class ConvNeXtImpl : public torch::nn::Module {

        public:
            ConvNeXtImpl(const std::vector<int>& depths, const std::vector<int>& dims, int num_classes = 1000, int image_channels = 3, float drop_path_rate = 0.1, float layer_scale_init_value = 1e-6);
            torch::Tensor forward(torch::Tensor x); 

        private:
            torch::nn::Conv2d stem{nullptr};
            torch::nn::Sequential stage1, stage2, stage3, stage4;
            Downsample downsample{nullptr};
            torch::nn::Sequential make_stage(int in_channels, int out_channels, int num_blocks,float drop_path_rate = 0.0f, float layer_scale = 1e-6);

    };
    TORCH_MODULE(ConvNeXt);
    class MobileViTImpl : public torch::nn::Module {
        public:
            MobileViTImpl(int img_size, std::vector<int> features_list, std::vector<int> d_list,
                          std::vector<int> transformer_depth, int expansion);
        
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::Sequential stem{nullptr};    // Initial Convolution + InvertedResidual
            torch::nn::Sequential stage1{nullptr};  // Stage 1
            torch::nn::Sequential stage2{nullptr};  // Stage 2 (MobileVitBlock)
            torch::nn::Sequential stage3{nullptr};  // Stage 3 (MobileVitBlock)
            torch::nn::Sequential stage4{nullptr};  // Stage 4 (MobileVitBlock + Conv)
    };
    TORCH_MODULE(MobileViT);
}

#endif // ARCHITECTURE_HPP