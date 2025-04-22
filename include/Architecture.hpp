#pragma once

#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "Backbone.hpp"


using namespace backbone;

namespace architecture {
    class ConvNextImpl : public torch::nn::Module {

        public:
            ConvNextImpl(int num_classes, const std::vector<int>& features, const int image_channel, bool down_sample);
            torch::Tensor forward(torch::Tensor x); 

        private:
            bool down_sample;
            torch::nn::Conv2d stem{nullptr};
            torch::nn::Sequential stage1, stage2, stage3, stage4;
            torch::nn::Sequential make_stage(int in_channels, int out_channels, int num_blocks);

    };
    TORCH_MODULE(ConvNext);
}

#endif // ARCHITECTURE_HPP