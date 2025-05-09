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

    /**
     * @struct ModelArgs
     * @brief Structure to hold different types of arguments for various model modes
     * 
     * This structure stores all possible parameters that can be passed to different
     * model types. It includes a type indicator to identify which model is being used.
    */ 
    struct ModelArgs {
        fs::path save_path;

        // Model 1 parameters, mostly used for Mnist dataset
        fs::path dataset_path;
        
        // Model 2 parameters, mostly used for the Oxford Pet Dataset
        fs::path image_path;
        fs::path label_path;
        
        /**
         * @enum ModelType
         * @brief Enumeration of supported model types
        */
        enum ModelType {        
            DATASET_MODEL,    
            IMAGE_LABEL_MODEL 
        } model_type;         // The type of model being used 
        
        /**
         * @enum NeuralNetType
         * @brief Enumeration of supported neural network architectures
        */
        enum NeuralNetType {      
            CNN,              
            CONVNEXT,         
            MOBILEVIT         
        } net_type;           // The type of neural network architecture to use 
        
        bool binary_mode; // Flag indicating whether to use binary classification mode 
    };

    /**
     * @brief Parse command line arguments for different model types
     * 
     * This function processes command line arguments to determine which model type
     * is being used and extracts the appropriate paths for that model.
     * 
     * @param argc Number of command line arguments
     * @param argv Array of command line argument strings
     * @return ModelArgs Structure containing parsed arguments and model type
     * @throws Exits program with status code 1 if arguments are invalid
    */
    ModelArgs parseArguments(int argc, char* argv[]);

    enum Mode { Test, Train };
    /**
    * @brief Retrieves the total number of parameters in a given PyTorch model.
    * 
    * @param model A reference to the PyTorch module.
    * @return int64_t Total number of parameters.
    */
    int64_t get_num_parameters(torch::nn::Module &model);
    /**
    * @brief Checks if the file at the given path is corrupt or unreadable.
    * 
    * @param path Path to the file.
    * @return true If the file is corrupt.
    * @return false If the file is valid.
    */
    bool is_corrupt(const std::string& path);
    /**
    * @brief Reverts image normalization by applying inverse transformation.
    * 
    * @param img The normalized image tensor.
    * @return torch::Tensor Unnormalized image tensor.
    */
    torch::Tensor unnormalize(const torch::Tensor& img);
    /**
    * @brief Verifies whether a directory contains corrupted images.
    * 
    * @param path Reference to the directory path to scan for corrupt images.
    */
    void verify_corrupt_images_dir(fs::path& path);
    /**
    * @brief Computes class weights for handling class imbalance during training.
    * 
    * @param labels A vector of integer class labels.
    * @return torch::Tensor A tensor of computed class weights.
    */
    torch::Tensor compute_class_weights(const std::vector<int>& labels);
    /**
    * @brief Visualizes a single image and label from a batch of tensors.
    * 
    * @param batch_images A tensor of batch images.
    * @param batch_labels A tensor of batch labels.
    * @param index Index of the image to visualize from the batch.
    */
    void visualizeBatchImage(const torch::Tensor& batch_images, const torch::Tensor& batch_labels, int index);
    /**
    * @brief Loads image file paths and corresponding labels from a dataset.
    * 
    * @param image_dir Directory containing the images.
    * @param annotation_path Path to the annotations (e.g., labels).
    * @param binary Boolean flag indicating whether it's binary classification.
    * @return std::tuple of shared pointers to vectors: image paths and labels.
    */
    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<int>> > get_image_path_and_labels(fs::path image_dir, fs::path annotation_path, bool binary);
    /**
    * @brief Splits the dataset into training and testing sets.
    * 
    * @param image_paths Shared pointer to vector of image file paths.
    * @param labels Shared pointer to vector of labels.
    * @param test_size Proportion of the dataset to include in the test split.
    * @return std::tuple containing shared pointers to train/test image paths and labels.
    */
    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<std::string>>, 
               std::shared_ptr<std::vector<int>>, std::shared_ptr<std::vector<int>> > train_test_split(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels  ,float test_size = 0.2f);
    /**
    * @brief Converts an OpenCV image to a Torch tensor.
    * 
    * @param image Reference to the OpenCV Mat image.
    * @return torch::Tensor Converted tensor.
    */
    torch::Tensor CVtoTensor(cv::Mat& image);
    /**
    * @brief Converts a Torch tensor back to an OpenCV image.
    * 
    * @param tensor_image Reference to the tensor image.
    * @return cv::Mat Converted OpenCV Mat image.
    */
    cv::Mat TensortoCV(torch::Tensor& tensor_image);
    /**
    * @brief Struct to accumulate precision, recall, and F1-score across multiple batches or evaluations.
    */
    struct MetricsAccumulator {
        float total_precision = 0.0;
        float total_recall = 0.0;
        float total_f1 = 0.0;
        int count = 0;
        /**
        * @brief Adds precision, recall, and F1-score of a single batch to the accumulator.
        * 
        * @param precision Precision value for the batch.
        * @param recall Recall value for the batch.
        * @param f1 F1-score for the batch.
        */
        void add(float precision, float recall, float f1) {
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            count += 1;
        }
        /**
        * @brief Computes the average precision, recall, and F1-score.
        * 
        * @return std::tuple<float, float, float> Average values (precision, recall, F1-score).
        */
        std::tuple<float, float, float> average() const {
            if (count == 0) return {0.0, 0.0, 0.0};
            return {
                total_precision / count,
                total_recall / count,
                total_f1 / count
            };
        }
    };
    /**
    * @brief Computes precision, recall, and F1-score for a batch of predictions in a multi-class classification setting.
    * 
    * @param targets Ground truth labels.
    * @param preds Predicted labels.
    * @param num_classes Total number of classes.
    * @return std::tuple<float, float, float> Tuple containing precision, recall, and F1-score.
    */
    std::tuple<float, float, float> compute_batch_metrics_multiclass(const std::vector<int>& targets, const std::vector<int>& preds, int num_classes); 
    /**
    * @brief Computes precision, recall, and F1-score for a batch of predictions in a binary classification setting.
    * 
    * @param targets Ground truth binary labels.
    * @param preds Predicted binary labels.
    * @return std::tuple<float, float, float> Tuple containing precision, recall, and F1-score.
    */
    std::tuple<float, float, float> compute_batch_metrics_binary(const std::vector<int>& targets, const std::vector<int>& preds);
}



#endif // UTILIS_HPP
