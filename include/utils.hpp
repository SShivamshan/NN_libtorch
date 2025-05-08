#pragma once

#ifndef UTILIS_HPP
#define UTILIS_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace utils{
    enum Mode { Test, Train };
    int64_t get_num_parameters(torch::nn::Module &model);
    bool is_corrupt(const std::string& path);
    torch::Tensor unnormalize(const torch::Tensor& img);
    void verify_corrupt_images_dir(fs::path& path);
    /*
    Used for computing the class weights for binary and multi class 
    */
   torch::Tensor compute_class_weights(const std::vector<int>& labels);
    void visualizeBatchImage(const torch::Tensor& batch_images, const torch::Tensor& batch_labels, int index);
    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<int>> > get_image_path_and_labels(fs::path& image_dir, fs::path& annotation_path, bool binary);
    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<std::string>>, 
               std::shared_ptr<std::vector<int>>, std::shared_ptr<std::vector<int>> > train_test_split(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels  ,float test_size = 0.2f);
    torch::Tensor CVtoTensor(cv::Mat& image);
    cv::Mat TensortoCV(torch::Tensor& tensor_image);
    struct MetricsAccumulator {
        float total_precision = 0.0;
        float total_recall = 0.0;
        float total_f1 = 0.0;
        int count = 0;
    
        void add(float precision, float recall, float f1) {
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            count += 1;
        }
    
        std::tuple<float, float, float> average() const {
            if (count == 0) return {0.0, 0.0, 0.0};
            return {
                total_precision / count,
                total_recall / count,
                total_f1 / count
            };
        }
    };
    std::tuple<float, float, float> compute_batch_metrics_multiclass(const std::vector<int>& targets, const std::vector<int>& preds, int num_classes); // 
    std::tuple<float, float, float> compute_batch_metrics_binary(const std::vector<int>& targets, const std::vector<int>& preds);
    // to load model in c++
}



#endif // UTILIS_HPP