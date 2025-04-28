#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include "Augmentation.hpp"

using namespace utils;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

    private:
        Mode mode_;
        std::shared_ptr<std::vector<std::string>> image_paths;
        std::shared_ptr<std::vector<int>> labels;
        std::shared_ptr<Augmentations> augmentations_;
        void set_augmentations();
    public:
        CustomDataset(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels, Mode mode_);
        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;
    
};

#endif // DATASET_HPP