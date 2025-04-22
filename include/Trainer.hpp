#pragma once

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>

template <typename ModelType>
class Trainer {
private:
    torch::optim::Optimizer* optimizer;
    int num_epochs;
    torch::Device device;
    bool save_model;
    boost::filesystem::path save_path;
    size_t train_size_;

public:
    Trainer(torch::optim::Optimizer* optimizer, int num_epochs, torch::Device device, bool save_model,
        const boost::filesystem::path save_path,size_t train_size);

    template <typename TrainLoader, typename TestLoader>
    std::map<std::string, std::vector<float>> fit(ModelType& model, TrainLoader& train_loader, TestLoader& test_loader);

    template <typename Dataloader>
    std::tuple<float, float> train(ModelType& model, Dataloader& train_loader, int epoch);

    template <typename Dataloader>
    std::tuple<float, float> test(ModelType& model, Dataloader& test_loader);

    void save(ModelType& model);

    ~Trainer();
};

#include "Trainer.tpp"



#endif // TRAINER_HPP