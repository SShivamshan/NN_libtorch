#pragma once

#ifndef BACKBONE_HPP
#define BACKBONE_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>

namespace backbone
{
    class ConvNeXtBlockImpl : public torch::nn::Module {
        public:
            ConvNeXtBlockImpl(int in_channels, float drop_path = 0.0, float layer_scale_init_value = 1e-6);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::Conv2d dwconv{nullptr};
            torch::nn::LayerNorm norm{nullptr};
            torch::nn::Linear pwconv1{nullptr}, pwconv2{nullptr};
            torch::Tensor gamma;

            // Drop Path parameters
            bool drop_path_enabled = false;
            float drop_path_rate;
            torch::Tensor drop_path(torch::Tensor x, float drop_prob);
    };
    TORCH_MODULE(ConvNeXtBlock);

    // MobileNetV2 block implementation
} // namespace backbone


#endif // BACKBONE_HPP