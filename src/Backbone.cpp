#include "Backbone.hpp"
using namespace backbone;

// https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
// https://tech.bertelsmann.com/en/blog/articles/convnext 

// ---------------------------------------------------------------- ConvNeXt ---------------------------------------------------------------- // 
ConvNeXtBlockImpl::ConvNeXtBlockImpl::ConvNeXtBlockImpl(int in_channels, float drop_path, float layer_scale_init_value):
    dwconv(torch::nn::Conv2dOptions(in_channels, in_channels, 7).padding(3).groups(in_channels)),
    norm(torch::nn::LayerNormOptions({in_channels}).elementwise_affine(true)),
    pwconv1(in_channels, 4 * in_channels),
    pwconv2(4 * in_channels, in_channels),
    drop_path_rate(drop_path)
{
    register_module("dwconv", dwconv);
    register_module("norm", norm);
    register_module("pwconv1", pwconv1);
    register_module("pwconv2", pwconv2);

    // Initialize gamma (Layer Scale)
    if (layer_scale_init_value > 0) {
        gamma = register_parameter("gamma", torch::full({in_channels}, layer_scale_init_value));
    }

    // DropPath is only applied during training
    if (drop_path > 0) {
        drop_path_enabled = true;
    }
}

torch::Tensor ConvNeXtBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = dwconv->forward(x);
    x = x.permute({0, 2, 3, 1}); // (N, C, H, W) -> (N, H, W, C)
    x = norm->forward(x);
    x = pwconv1->forward(x);
    x = torch::gelu(x);
    x = pwconv2->forward(x);
    if (gamma.defined()) {
        x = x * gamma;
    }
    x = x.permute({0, 3, 1, 2}); // (N, H, W, C) -> (N, C, H, W)
    // Apply DropPath
    if (drop_path_enabled && is_training()) {
        x = drop_path(x, drop_path_rate);
    }
    x = residual + x;
    return x;
}
// Solution adapted from : https://github.com/FrancescoSaverioZuppichini/DropPath 
torch::Tensor ConvNeXtBlockImpl::drop_path(torch::Tensor x, float drop_prob){
    if (drop_prob == 0.0 || !x.requires_grad()) {
        return x;
    }

    torch::Tensor keep_prob = torch::tensor(1.0 - drop_prob, torch::kFloat).to(x.device());
    torch::Tensor mask = torch::rand_like(x) < keep_prob;
    return x * mask / keep_prob;  // Scale to maintain expected value

}