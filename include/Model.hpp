#pragma once

#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "Architecture.hpp"
using namespace architecture;

namespace Model
{

    /**
     * @brief A simple feedforward neural network for MNIST classification.
     * 
     * Architecture:
     * - Input: Flattened 28x28 image (784)
     * - Hidden layers: 512 → 256 → 128 → 64
     * - Output: 10 neurons (for digits 0–9)
    */
    class ClassifierImpl: public torch::nn::Module{
        public:
            /**
             * @brief Constructs the MNIST classifier.
             * 
             * @param image_size Input image size.
             * @param in_channels Vector defining the width of each layer, default : [512,256,128,64,10]
             * @param num_classes Number of output classes, default : 1.
            */
            ClassifierImpl(const size_t image_size, std::vector<int> in_channels = {512, 256, 128, 64, 10},const int num_classes = 10);
            /**
             * @brief Forward pass of the classifier.
             * 
             * @param x Input tensor (batch_size x image_size).
             * @return Output tensor (batch_size x num_classes).
            */
            torch::Tensor forward(torch::Tensor x);
        private:
            torch::nn::Linear fc1{nullptr},fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
            torch::nn::Dropout dropout;
    };
    TORCH_MODULE(Classifier);

    class CNNClassifierImpl: public torch::nn::Module{
        public:
            CNNClassifierImpl(std::vector<int> in_channels = {64,128,256,512,1024}, const int num_classes = 10);
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Sequential make_cnn_blocks(int in_channels, int out_channels,int stride, int padding, const std::string& block_name);
            torch::nn::Sequential cnn_block1{nullptr},cnn_block2{nullptr},cnn_block3{nullptr},cnn_block4{nullptr},cnn_block5{nullptr};
            torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
            torch::nn::MaxPool2d max_pool{nullptr};
            torch::nn::Linear _fc{nullptr};
            torch::nn::BatchNorm1d _batchnorm{nullptr};  
            torch::nn::Linear _classifier{nullptr};
            torch::nn::Dropout _dropout1{nullptr};
            torch::nn::Dropout _dropout2{nullptr};
    };
    TORCH_MODULE(CNNClassifier);

    /**
     * @brief ConvNeXt-based image classifier.
     * 
     * Uses a ConvNeXt backbone with custom classifier head. Inspired by:
     * - https://github.com/facebookresearch/ConvNeXt
     * - https://tech.bertelsmann.com/en/blog/articles/convnext
     * 
     * Architecture:
     * - Backbone: ConvNeXt with configurable width and depth.
     * - Pooling: AdaptiveAvgPool2d.
     * - Classification head: Linear → LayerNorm → GelU + Dropout → Linear
    */
    class ConvNextClassifierImpl: public torch::nn::Module {
        public:
            /**
             * @brief Constructs the ConvNeXt classifier.
             * 
             * @param num_classes Number of output classes.
             * @param image_channel Number of input channels (e.g., 3 for RGB).
             * @param use_tiny Whether to use ConvNeXt-Tiny configuration.
            */
            ConvNextClassifierImpl(int num_classes, const int image_channel = 3, bool  use_tiny = false);
            /**
             * @brief Forward pass through the ConvNeXt classifier.
             * 
             * @param x Input tensor (batch_size x channels x height x width).
             * @return Output logits tensor (batch_size x num_classes).
            */
            torch::Tensor forward(torch::Tensor x);

        private:
            ConvNeXt convnext{nullptr};
            torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
            torch::nn::LayerNorm norm{nullptr};
            torch::nn::Linear fc{nullptr};
            torch::nn::Dropout dropout1{nullptr};
            torch::nn::Dropout dropout2{nullptr};
            torch::nn::Linear classifier{nullptr};
    };
    TORCH_MODULE(ConvNextClassifier);

    /**
     * @brief MobileViT-based image classifier.
     * 
     * Uses a MobileViT backbone and a classifier head. Inspired by:
     * - https://github.com/wilile26811249/MobileViT/blob/main/models/module.py   
     * 
     * Combines MobileNet-like convolutions with lightweight transformers (ViT blocks).
    */
    class MobileViTClassifierImpl: public torch::nn::Module{
        public:
            /**
             * @brief Constructs a MobileViT classifier.
             * 
             * @param features_list List of feature map sizes at each stage.
             * @param d_list List of depth values for convolutional blocks.
             * @param transformer_depth Depth of transformer layers at each stage.
             * @param expansion Expansion ratio for convolution blocks.
             * @param num_classes Number of output classes.
            */
            MobileViTClassifierImpl(std::vector<int> features_list, std::vector<int> d_list, std::vector<int> transformer_depth, int expansion,int num_classes);
            /**
             * @brief Forward pass through the MobileViT classifier.
             * 
             * @param x Input tensor (batch_size x channels x height x width).
             * @return Output tensor (batch_size x num_classes).
            */
            torch::Tensor forward(torch::Tensor x);
        
        private:
            MobileViT _mobilevit{nullptr};
            torch::nn::AdaptiveAvgPool2d _avgpool{nullptr};
            torch::nn::Linear _fc{nullptr};
            torch::nn::BatchNorm1d _batchnorm{nullptr};  
            torch::nn::Linear _classifier{nullptr};
            torch::nn::Dropout _dropout{nullptr};
    };
    TORCH_MODULE(MobileViTClassifier);

} // namespace Model


#endif // MODEL_HPP
