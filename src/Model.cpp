#include "Model.hpp"
using namespace Model;


// ---------------------------------------------------------------- MobileViT Classifier ---------------------------------------------------------------- //  
MobileViTClassifierImpl::MobileViTClassifierImpl(std::vector<int> features_list, std::vector<int> d_list,
    std::vector<int> transformer_depth, int expansion,int num_classes){
    
    _mobilevit = register_module("MobileViT", MobileViT(features_list, d_list, transformer_depth, expansion));
    int convnext_output_channels = features_list.back(); 
    _fc = register_module("FC", torch::nn::Linear(convnext_output_channels, 512));
    _batchnorm = register_module("BatchNorm", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)));
    _dropout = register_module("Dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.25)));
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

// ---------------------------------------------------------------- ConvNeXt Classifier ---------------------------------------------------------------- // 

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

}

torch::Tensor ConvNextClassifierImpl::forward(torch::Tensor x)
{
    x = convnext->forward(x);                  // (B, C, H, W)
    x = x.permute({0, 2, 3, 1}).contiguous();  // (B, H, W, C)
    x = norm->forward(x);                    
    x = x.permute({0, 3, 1, 2}).contiguous();  // (B, C, H, W)

    x = avgpool->forward(x);                   // (B, C, 1, 1)
    x = x.flatten(1);                          // (B, C)
    x = dropout1->forward(x);
    x = fc->forward(x);
    x = torch::gelu(x);                        
    x = dropout2->forward(x);
    x = classifier->forward(x);
    return x;      
}

// ---------------------------------------------------------------- Linear Classifier ---------------------------------------------------------------- // 
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

// ---------------------------------------------------------------- CNN Classifier ---------------------------------------------------------------- // 

CNNClassifierImpl::CNNClassifierImpl(std::vector<int> channels, const int num_classes) {
    TORCH_CHECK(channels.size() == 5, "Expected 5 elements in channels vector");

    cnn_block1 = register_module("cnn_block1", make_cnn_blocks(3, channels[0], 1, 1, "block1"));
    cnn_block2 = register_module("cnn_block2", make_cnn_blocks(channels[0], channels[1], 1, 1, "block2"));
    cnn_block3 = register_module("cnn_block3", make_cnn_blocks(channels[1], channels[2], 1, 1, "block3"));
    cnn_block4 = register_module("cnn_block4", make_cnn_blocks(channels[2], channels[3], 1, 1, "block4"));
    cnn_block5 = register_module("cnn_block5", make_cnn_blocks(channels[3], channels[4], 1, 1, "block5"));

    avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7, 7})));
    max_pool = register_module("max_pool",torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    _fc = register_module("_fc", torch::nn::Linear(channels[4]*7*7, 512));
    _batchnorm = register_module("_batchnorm", torch::nn::BatchNorm1d(512));
    _dropout1 = register_module("_dropout1", torch::nn::Dropout(0.25));
    _dropout2 = register_module("_dropout2", torch::nn::Dropout(0.3));
    _classifier = register_module("_classifier", torch::nn::Linear(512, num_classes));
}

torch::nn::Sequential CNNClassifierImpl::make_cnn_blocks(int in_channels, int out_channels, int stride, int padding, const std::string& block_name) {
    torch::nn::Sequential block;

    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                    .stride(stride).padding(padding).bias(false));
    auto bn = torch::nn::BatchNorm2d(out_channels);
    auto relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));

    block->push_back(register_module(block_name + "_conv", conv));
    block->push_back(register_module(block_name + "_bn", bn));
    block->push_back(relu);

    return block;
}

torch::Tensor CNNClassifierImpl::forward(torch::Tensor x) {
    x = cnn_block1->forward(x);
    x = max_pool->forward(x);
    x = _dropout1->forward(x);
    x = cnn_block2->forward(x);
    x = max_pool->forward(x);
    x = _dropout1->forward(x);
    x = cnn_block3->forward(x);
    x = max_pool->forward(x);
    x = _dropout1->forward(x);
    x = cnn_block4->forward(x);
    x = max_pool->forward(x);
    x = _dropout1->forward(x);
    x = cnn_block5->forward(x);

    x = avgpool->forward(x);
    x = _dropout1->forward(x);
    x = x.view({x.size(0), -1});  
    x = _fc->forward(x);
    x = _batchnorm->forward(x);
    x = torch::relu(x);
    x = _dropout2->forward(x);
    x = _classifier->forward(x);
    return x;
}
