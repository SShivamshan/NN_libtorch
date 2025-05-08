#include "Model.hpp"
using namespace Model;


MobileViTClassifierImpl::MobileViTClassifierImpl(int img_size, std::vector<int> features_list, std::vector<int> d_list,
    std::vector<int> transformer_depth, int expansion,int num_classes){
    
    _mobilevit = register_module("MobileViT", MobileViT(img_size,features_list, d_list, transformer_depth, expansion));
    int convnext_output_channels = features_list.back() * 7 * 7; 
    _fc = register_module("FC", torch::nn::Linear(convnext_output_channels, 512));
    _batchnorm = register_module("BatchNorm", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)));
    _dropout = register_module("Dropout1", torch::nn::Dropout(torch::nn::DropoutOptions(0.25)));
    _classifier = register_module("Classifier", torch::nn::Linear(512, num_classes));
}

torch::Tensor MobileViTClassifierImpl::forward(torch::Tensor x){
    auto feats = _mobilevit->forward(x);                   
    auto pooled = _avgpool(feats);                        
    // pooled = _dropout1(pooled);                        
    auto flattened = pooled.view({pooled.size(0), -1});   
    auto fc_output = _fc(flattened);                       
    fc_output = _batchnorm(fc_output);                           
    fc_output = _dropout(torch::relu(fc_output));                      
    return _classifier(fc_output);
}

ConvNextClassifierImpl::ConvNextClassifierImpl(int num_classes,const int image_channel,bool use_tiny)
{   
    std::vector<int> depths;
    std::vector<int> dims;
    float drop_path_rate;
    
    if (use_tiny) {
        // Tiny variant nearly ~30 millions parameters
        depths = {3, 3, 9, 3};
        dims = {96, 192, 384, 768};
        drop_path_rate = 0.1;
    } else {
        // Even smaller variant with 13 million parameters
        depths = {3, 3, 9, 3};
        dims = {64, 128, 256, 512};
        drop_path_rate = 0.2;
    }
    convnext = register_module("convnext", ConvNeXt(depths, dims, num_classes, image_channel, drop_path_rate));

    avgpool = register_module("AdaptiveAvgPool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    int final_dim = dims.back();
    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({final_dim})));
    fc = register_module("FC", torch::nn::Linear(final_dim, 512));
    dropout1 = register_module("Dropout1", torch::nn::Dropout(torch::nn::DropoutOptions(0.25)));
    dropout2 = register_module("Dropout2", torch::nn::Dropout(torch::nn::DropoutOptions(0.3)));
    classifier = register_module("Classifier", torch::nn::Linear(512, num_classes));

    // Initialize the classifier head
    // torch::nn::init::trunc_normal_(classifier->weight, 0.02);
    // torch::nn::init::zeros_(classifier->bias);
}

torch::Tensor ConvNextClassifierImpl::forward(torch::Tensor x)
{
    x = convnext->forward(x);                  // (B, C, H, W)
    x = x.permute({0, 2, 3, 1}).contiguous();  // (B, H, W, C)
    x = norm->forward(x);                    
    x = x.permute({0, 3, 1, 2}).contiguous();  // Back to (B, C, H, W)

    x = avgpool->forward(x);                   // (B, C, 1, 1)
    x = x.flatten(1);                          // (B, C)
    x = dropout1->forward(x);
    x = fc->forward(x);
    x = torch::gelu(x);                        // GELU non-linearity
    x = dropout2->forward(x);
    x = classifier->forward(x);
    return x;      
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