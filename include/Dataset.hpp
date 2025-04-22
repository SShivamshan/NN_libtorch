#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <iostream>
#include <vector>


class CustomDataset : public torch::data::datasets::Dataset {
    
}

#endif // DATASET_HPP