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
    class ConvNormActImpl : public torch::nn::Module {
        public:
            ConvNormActImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, 
                            int padding = -1, int groups = 1, bool use_bn = true, bool use_act = true);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::Conv2d conv{nullptr};
            torch::nn::BatchNorm2d bn{nullptr};
            torch::nn::SiLU act{nullptr};
            bool use_bn;
            bool use_act;
    };
    TORCH_MODULE(ConvNormAct);

    class MultiHeadSelfAttentionImpl : public torch::nn::Module {
        public:
        MultiHeadSelfAttentionImpl(int64_t dim, int64_t num_heads);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            int64_t num_heads;
            int64_t dim_head;
            torch::nn::Linear to_qvk{nullptr};
            torch::nn::Linear w_out{nullptr};
            float scale_factor;
    };
    TORCH_MODULE(MultiHeadSelfAttention);

    class FFNImpl : public torch::nn::Module {
        public:
            FFNImpl(int64_t dim, int64_t hidden_dim, double dropout = 0.0);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::Sequential net;
    };
    TORCH_MODULE(FFN);
        
    class TransformerImpl : public torch::nn::Module {
        public:
            TransformerImpl(int dim, int depth, int heads, int dim_head, int mlp_dim, double dropout = 0.1);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::ModuleList layers;
    };
    TORCH_MODULE(Transformer);
        
    class InvertedResidualImpl : public torch::nn::Module {
        public:
            InvertedResidualImpl(int in_channels, int out_channels, int stride = 1, float expand_ratio = 2.0);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            bool use_res_connect;
            torch::nn::Sequential conv;
    };
    TORCH_MODULE(InvertedResidual);
        
    class MobileVitBlockImpl : public torch::nn::Module {
        public:
            MobileVitBlockImpl(int in_channels, int out_channels, int d_model, int layers, int mlp_dim);
            torch::Tensor forward(torch::Tensor x);
        
        private:
            torch::nn::Sequential local_representation{nullptr};  // Using torch::nn::Sequential for stacking layers
            Transformer transformer{nullptr};                     // Register Transformer module
            torch::nn::Conv2d fusion_block1{nullptr};             // Conv layer for fusion
            torch::nn::Conv2d fusion_block2{nullptr};             // Conv layer for output fusion
    };
    TORCH_MODULE(MobileVitBlock);
} // namespace backbone


#endif // BACKBONE_HPP