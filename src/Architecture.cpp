#include "Architecture.hpp"

using namespace architecture;

// https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

// ---------------------------------------------------------------- ConvNeXt ---------------------------------------------------------------- //  
ConvNextImpl::ConvNextImpl(int num_classes, const std::vector<int>& features, const int image_channel, bool down_sample) {
    down_sample = down_sample;
    // Expecting features to be something like: {64, 128, 256, 512, 1024}
    TORCH_CHECK(features.size() == 5, "Expected 5 elements in features vector");
    // Patch Embedding (Stem Conv)
    stem = register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(image_channel, features[0], 4).stride(4).padding(0)));

    // Stages
    stage1 = register_module("stage1", make_stage(features[0], features[1], 3));
    stage2 = register_module("stage2", make_stage(features[1], features[2], 3));
    stage3 = register_module("stage3", make_stage(features[2], features[3], 9));
    stage4 = register_module("stage4", make_stage(features[3], features[4], 3));

}

torch::Tensor ConvNextImpl::forward(torch::Tensor x) {

    
    x = stem->forward(x);
    
    auto feat1 = stage1->forward(x);
    auto feat2 = stage2->forward(feat1);
    auto feat3 = stage3->forward(feat2);
    auto feat4 = stage4->forward(feat3);

    return feat4;
}


torch::nn::Sequential ConvNextImpl::make_stage(int in_channels, int out_channels, int num_blocks) {
    torch::nn::Sequential stage;
    if (down_sample){
        stage->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 2).stride(2)));  // Downsample
    }else{
        stage->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
    }
    for (int i = 0; i < num_blocks; i++) {
        stage->push_back(ConvNeXtBlock(out_channels, out_channels * 4)); 
    }
    return stage;
}