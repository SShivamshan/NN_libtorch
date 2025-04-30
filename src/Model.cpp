#include "Model.hpp"
using namespace Model;

ConvNextClassifierImpl::ConvNextClassifierImpl(int num_classes,
    std::vector<int> in_channels,const int image_channel,bool down_sample)
{
    _convnext = register_module("ConvNext", ConvNext(num_classes, in_channels, image_channel, down_sample));

    _avgpool = register_module("AdaptiveAvgPool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    _dropout = register_module("Dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.3)));
    int convnext_output_channels = in_channels.back(); // assuming final stage output
    _classifier = register_module("Classifier", torch::nn::Linear(convnext_output_channels, num_classes));
}

torch::Tensor ConvNextClassifierImpl::forward(torch::Tensor x)
{
    auto feats = _convnext->forward(x); // (B, C, H, W)
    auto pooled = _avgpool(feats);      // (B, C, 1, 1)
    auto flattened = pooled.view({pooled.size(0), -1}); // (B, C)
    flattened = _dropout(flattened);
    return _classifier(flattened); // (B, num_classes)
}


ClassifierImpl::ClassifierImpl(const size_t image_size, std::vector<int> in_channels, const int num_classes)
    : dropout(0.2)
{
    size_t input_dim = image_size * image_size;

    fc1 = register_module("fc1", torch::nn::Linear(input_dim, in_channels[0]));
    fc2 = register_module("fc2", torch::nn::Linear(in_channels[0], in_channels[1]));
    fc3 = register_module("fc3", torch::nn::Linear(in_channels[1], in_channels[2]));
    fc4 = register_module("fc4", torch::nn::Linear(in_channels[2], in_channels[3]));
    fc5 = register_module("fc5", torch::nn::Linear(in_channels[3], num_classes));

    register_module("dropout", dropout);
}

torch::Tensor ClassifierImpl::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    x = dropout(torch::relu(fc1->forward(x)));
    x = dropout(torch::relu(fc2->forward(x)));
    x = dropout(torch::relu(fc3->forward(x)));
    x = dropout(torch::relu(fc4->forward(x)));
    x = fc5->forward(x); 
    return x;
}