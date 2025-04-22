#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <boost/filesystem.hpp>

#include "Model.hpp"
#include "utils.hpp"
#include "Trainer.hpp"

using namespace Model;
using namespace utils;
namespace fs = boost::filesystem;

// - FInish utilis.cpp
// - To do finish Creating dataset class for Oxford Dataset
// - Train the ConvNext and MobileVit
// - Try pruninig or quantization
struct Mnist_Classifier
{
    fs::path model_save_path;
    const int64_t num_classes = 10;
    std::vector<int> feature_dims = {512,256,128,64};
    const int64_t batch_size = 256;
    const size_t num_epochs = 10;
    const double learning_rate = 0.0015; 
    
    size_t num_train_samples;
    

    void run(fs::path dataset_path, fs::path model_save_path, torch::Device device){
        auto train_dataset = torch::data::datasets::MNIST(dataset_path.string()) // this alone returns as torch::Example(data,target)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // adding this changes the type
        .map(torch::data::transforms::Stack<>());

        auto test_dataset = torch::data::datasets::MNIST(dataset_path.string(),torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

        num_train_samples = (int)train_dataset.size().value();
        
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), batch_size);

        auto batch_iter = train_loader->begin(); // iterator 
        auto batch = *batch_iter; // gives out [64,1,28,28] so we derefernce first or (*batch_iter).data[0]

        // Access image and label from the batch
        auto image = batch.data[0];   // shape: [1, 28, 28]
        auto label = batch.target[0]; // shape: [1]

        const int image_channel = image.sizes()[1];
        Classifier model = Classifier(image_channel,feature_dims,num_classes);
        model->to(device);
        // torch::optim::SGD optimizer(model->parameters(),torch::optim::SGDOptions(learning_rate));
        torch::optim::Adam optimizer(model->parameters(), /*lr=*/learning_rate);

        Trainer<Classifier> trainer(&optimizer, num_epochs, device, true, model_save_path,num_train_samples);
        
        std::map<std::string, std::vector<float>> history = trainer.fit(model, *train_loader, *test_loader);

        std::cout << "Training completed!" << std::endl;
        std::cout << "Final training loss: " << history["train_loss"].back() << std::endl;
        std::cout << "Final training accuracy: " << history["train_accu"].back() << std::endl;
        std::cout << "Final test loss: " << history["test_loss"].back() << std::endl;
        std::cout << "Final test accuracy: " << history["test_accu"].back() << std::endl;
    }

};


int main(int argc, char* argv[]) {
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mnist_data_path> <save_path>" << std::endl;
        return 1;
    }
    
    // Resolve dataset path
    fs::path dataset_path;
    if (fs::path(argv[1]).is_absolute()) {
        dataset_path = argv[1];
    } else {
        dataset_path = fs::current_path() / argv[1];
    }

    // Resolve save path
    fs::path save_path;
    if (fs::path(argv[2]).is_absolute()) {
        save_path = argv[2];
    } else {
        save_path = fs::current_path() / argv[2];
    }

    std::cout << "Dataset Path: " << dataset_path << std::endl;
    std::cout << "Save Path: " << save_path << std::endl;

    if (!fs::exists(dataset_path)) {
        std::cerr << "Error: Dataset path does not exist: " << dataset_path << std::endl;
        return 1;
    }

    torch::manual_seed(1);
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Mnist_Classifier classifier;
    classifier.run(dataset_path,save_path,device);

      
    return 0;
}




