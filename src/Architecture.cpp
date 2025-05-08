#include "Architecture.hpp"

using namespace architecture;

// https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

// ---------------------------------------------------------------- ConvNeXt ---------------------------------------------------------------- //  
ConvNeXtImpl::ConvNeXtImpl(const std::vector<int>& depths, const std::vector<int>& dims, int num_classes, int image_channels, float drop_path_rate, float layer_scale_init_value) {
    // Expecting features to be something like: {64, 128, 256, 512}
    TORCH_CHECK(depths.size() == 4, "Expected 4 elements in depths vector");
    TORCH_CHECK(dims.size() == 4, "Expected 4 elements in dims vector");
    // Patch Embedding (Stem Conv)
    stem = register_module("stem", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(image_channels, dims[0], 4).stride(4).padding(0).bias(false)
    ));
    
    torch::nn::init::kaiming_normal_(stem->weight, 0.0, torch::kFanOut);
    
    stage1 = register_module("stage1", make_stage(dims[0], dims[0], depths[0], drop_path_rate, layer_scale_init_value));
    stage2 = register_module("stage2", make_stage(dims[0], dims[1], depths[1], drop_path_rate, layer_scale_init_value));
    stage3 = register_module("stage3", make_stage(dims[1], dims[2], depths[2], drop_path_rate, layer_scale_init_value));
    stage4 = register_module("stage4", make_stage(dims[2], dims[3], depths[3], drop_path_rate, layer_scale_init_value));

}

torch::Tensor ConvNeXtImpl::forward(torch::Tensor x) {
    x = stem->forward(x);
    
    auto feat1 = stage1->forward(x);
    auto feat2 = stage2->forward(feat1);
    auto feat3 = stage3->forward(feat2);
    auto feat4 = stage4->forward(feat3);

    return feat4;
}

torch::nn::Sequential ConvNeXtImpl::make_stage(int in_channels, int out_channels, int num_blocks, float drop_path_rate, float layer_scale) {
    torch::nn::Sequential stage;
    if (in_channels != out_channels) {
        // downsample = register_module("downsample", Downsample(in_channels, out_channels, 2));
        std::string downsample_name = "downsample_" + std::to_string(out_channels);
        auto downsample_block = register_module(downsample_name, Downsample(in_channels, out_channels, 2));
        stage->push_back(downsample_block);
    }

    for (int i = 0; i < num_blocks; i++) {
        // stage->push_back(ConvNeXtBlock(out_channels, out_channels * 4)); 
        float block_drop_path = drop_path_rate * i / (num_blocks - 1.0);
        stage->push_back(ConvNeXtBlock(out_channels, block_drop_path, layer_scale));
    }
    return stage;
}

DownsampleImpl::DownsampleImpl(int in_channels, int out_channels, int stride) :
    norm(torch::nn::LayerNormOptions({in_channels})),
    conv(torch::nn::Conv2dOptions(in_channels, out_channels, 2).stride(stride).padding(0).bias(false))
{
    register_module("norm", norm);
    register_module("conv", conv);
}

torch::Tensor DownsampleImpl::forward(torch::Tensor x) {
    x = x.permute({0, 2, 3, 1}).contiguous(); // Make it contiguous before norm
    x = norm->forward(x);
    x = x.permute({0, 3, 1, 2}).contiguous(); // Make it contiguous again
    x = conv->forward(x);
    return x;
}



// ---------------------------------------------------------------- MobileViT ---------------------------------------------------------------- //
MobileViTImpl::MobileViTImpl(int img_size, std::vector<int> features_list, std::vector<int> d_list,
    std::vector<int> transformer_depth, int expansion)
{
    // Stem Block
    stem = register_module("stem", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, features_list[0], 3).stride(2).padding(1)),
        InvertedResidual(features_list[0], features_list[1], 1, expansion)
    ));

    // Stage 1
    stage1 = register_module("stage1", torch::nn::Sequential(
        InvertedResidual(features_list[1], features_list[2], 2, expansion),
        InvertedResidual(features_list[2], features_list[2], 1, expansion),
        InvertedResidual(features_list[2], features_list[3], 1, expansion)
    ));

    // Stage 2
    stage2 = register_module("stage2", torch::nn::Sequential(
        InvertedResidual(features_list[3], features_list[4], 2, expansion),
        MobileVitBlock(features_list[4], features_list[5], d_list[0], transformer_depth[0], d_list[0] * 2)
    ));

    // Stage 3
    stage3 = register_module("stage3", torch::nn::Sequential(
        InvertedResidual(features_list[5], features_list[6], 2, expansion),
        MobileVitBlock(features_list[6], features_list[7], d_list[1], transformer_depth[1], d_list[1] * 4)
    ));

    // Stage 4
    stage4 = register_module("stage4", torch::nn::Sequential(
        InvertedResidual(features_list[7], features_list[8], 2, expansion),
        MobileVitBlock(features_list[8], features_list[9], d_list[2], transformer_depth[2], d_list[2] * 4),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(features_list[9], features_list[10], 1).stride(1).padding(0))
    ));
}

torch::Tensor MobileViTImpl::forward(torch::Tensor x) {
    x = stem->forward(x);
    x = stage1->forward(x);
    x = stage2->forward(x);
    x = stage3->forward(x);
    x = stage4->forward(x);

    return x;
}