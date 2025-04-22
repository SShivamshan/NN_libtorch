#include "utils.hpp"

namespace utils
{
    int64_t get_num_parameters(torch::nn::Module &model){
        int64_t total_params = 0;
        for(const auto &parameters: model.parameters()){
            total_params+=parameters.numel();
        }
        return total_params;
    }
    
} // namespace utils

