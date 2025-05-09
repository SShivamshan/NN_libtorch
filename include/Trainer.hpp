#pragma once

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <boost/filesystem.hpp>
#include "utils.hpp"

using namespace utils;
using namespace std::chrono;

/**
* @brief A generic training utility class for training and evaluating PyTorch models using LibTorch.
* 
* @tparam ModelType The model type, typically derived from torch::nn::Module.
*/
template <typename ModelType>
class Trainer {
private:
    torch::optim::Optimizer* optimizer;
    int num_epochs;
    torch::Device device;
    bool save_model;
    boost::filesystem::path save_path;
    size_t train_size_;
    bool binary_;
    bool oxford;
    int num_classes;
    MetricsAccumulator metrics_acc;
    torch::Tensor pos_weight;

public:
    /**
     * @brief Constructor made for the Mnist Dataset
     * 
     * @param optimizer Pointer to the optimizer.
     * @param num_epochs Number of training epochs.
     * @param device Device (CPU or CUDA) to train on.
     * @param save_model Whether to save the model after training.
     * @param save_path Filesystem path to save the trained model.
     * @param train_size Size of the training dataset.
     * @param binary Whether this is binary classification.
     * @param num_classes Total number of classes.
    */
    Trainer(torch::optim::Optimizer* optimizer, int num_epochs, torch::Device device, bool save_model,
        const boost::filesystem::path save_path,size_t train_size, bool binary,int num_classes,bool oxford);
    /**
     * @brief Constructor with class weights made for the Oxford Pet Dataset
     * 
     * @param optimizer Pointer to the optimizer.
     * @param num_epochs Number of training epochs.
     * @param device Device (CPU or CUDA) to train on.
     * @param save_model Whether to save the model after training.
     * @param save_path Filesystem path to save the trained model.
     * @param train_size Size of the training dataset.
     * @param binary Whether this is binary classification.
     * @param num_classes Total number of classes.
     * @param weights Class weighting tensor (e.g., for handling class imbalance).
    */
    Trainer(torch::optim::Optimizer* optimizer, int num_epochs, torch::Device device, bool save_model,
            const boost::filesystem::path save_path,size_t train_size, bool binary,int num_classes,torch::Tensor weights, bool oxford);
    
    /**
     * @brief Fits the model to the training data and evaluates it on the test data over multiple epochs.
     * 
     * @tparam TrainLoader Type of the training data loader.
     * @tparam TestLoader Type of the testing data loader.
     * @param model The model to be trained.
     * @param train_loader DataLoader for the training set.
     * @param test_loader DataLoader for the testing set.
     * @return A map of metric names to vectors of values over epochs.
    */
    template <typename TrainLoader, typename TestLoader>
    std::map<std::string, std::vector<float>> fit(ModelType& model, TrainLoader& train_loader, TestLoader& test_loader);
    
    /**
     * @brief Trains the model for a single epoch.
     * 
     * @tparam Dataloader Type of the training data loader.
     * @param model The model to be trained.
     * @param train_loader DataLoader for the training set.
     * @param epoch The current epoch number.
     * @return A tuple containing the average loss and accuracy.
    */
    template <typename Dataloader>
    std::tuple<float, float> train(ModelType& model, Dataloader& train_loader, int epoch);
    
    /**
     * @brief Evaluates the model on the test dataset.
     * 
     * @tparam Dataloader Type of the test data loader.
     * @param model The model to be evaluated.
     * @param test_loader DataLoader for the test set.
     * @return A tuple containing the average loss and accuracy.
    */
    template <typename Dataloader>
    std::tuple<float, float> test(ModelType& model, Dataloader& test_loader);
    
    /**
     * @brief Saves the model to the specified path.
     * 
     * @param model The model to save.
    */
    void save(ModelType& model);

    ~Trainer();
};

#include "Trainer.tpp"



#endif // TRAINER_HPP
