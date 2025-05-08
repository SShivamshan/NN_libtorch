#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <boost/filesystem.hpp>

#include "Model.hpp"
#include "utils.hpp"
#include "Trainer.hpp"
#include "Dataset.hpp"

using namespace Model;
using namespace utils;
namespace fs = boost::filesystem;

// - Load in python
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

        const int image_size = image.sizes()[1]; // image_size
        Classifier model = Classifier(image_size,feature_dims,num_classes);
        model->to(device);
        // torch::optim::SGD optimizer(model->parameters(),torch::optim::SGDOptions(learning_rate));
        torch::optim::Adam optimizer(model->parameters(), learning_rate);

        Trainer<Classifier> trainer(&optimizer, num_epochs, device, false, model_save_path,num_train_samples,false,num_classes);
        
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

    // Mnist_Classifier classifier;
    // classifier.run(dataset_path,save_path,device);

    // Hyperparameters
    const int64_t num_classes = 37;
    const int64_t batch_size = 32;
    const size_t num_epochs = 25;
    const double learning_rate = 1e-3; 

    fs::path image_dir = "/home/shiv/Desktop/Pytorch_project/data/Oxford_IIIT_PET/images";
    fs::path label_path = "/home/shiv/Desktop/Pytorch_project/data/Oxford_IIIT_PET/annotations/list.txt";

    auto data = get_image_path_and_labels(image_dir,label_path,true);

    auto image_paths = std::get<0>(data);
    auto labels = std::get<1>(data);
    torch::Tensor weights = compute_class_weights(*labels);
    // std::cout << weights << std::endl;
    auto [train_images, test_images, train_labels, test_labels] = train_test_split(image_paths, labels, 0.2f);
    auto train_dataset = (
        CustomDataset(train_images, train_labels, Mode::Train)
        .map(torch::data::transforms::Normalize<>(
            {0.485, 0.456, 0.406},
            {0.229, 0.224, 0.225}
        ))
        .map(torch::data::transforms::Stack<>())
    ); // this has a format of MapDataset(with CustomDataset, NOrmalize and stack)
    auto test_dataset = (CustomDataset(test_images, test_labels, Mode::Test)
        .map(torch::data::transforms::Normalize<>(
            {0.485, 0.456, 0.406},
            {0.229, 0.224, 0.225}
        ))
        .map(torch::data::transforms::Stack<>())
    );
    size_t num_train_samples = (int)train_dataset.size().value();
    // std::cout << num_train_samples << std::endl;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    auto batch_iter = train_loader->begin(); // iterator 
    auto batch = *batch_iter; // gives out [batch_size,1,height,width] so we derefernce first or (*batch_iter).data[0]
    // visualizeBatchImage(batch.data, batch.target, 0);
    // Access image and label from the batch
    auto image = batch.data[0];   // shape: [channel,hegiht,width]
    auto label = batch.target[0]; // shape: [1]

    const int image_size = image.sizes()[0];
    ConvNextClassifier model(num_classes,image_size,true);
    int64_t params = get_num_parameters(*model);
    std::cout << params << std::endl;
    model->to(device);
    
    // torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    // torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::optim::AdamWOptions options(learning_rate);
    options.weight_decay(0.001);  
    torch::optim::AdamW optimizer(model->parameters(), options);
        
    Trainer<ConvNextClassifier> trainer(&optimizer, num_epochs, device, true, save_path,num_train_samples,true,num_classes,weights);
    
    std::map<std::string, std::vector<float>> history = trainer.fit(model, *train_loader, *test_loader);
    std::cout << "Training completed!" << std::endl;
    std::cout << "Final training loss: " << history["train_loss"].back() << std::endl;
    std::cout << "Final training accuracy: " << history["train_accu"].back() << std::endl;
    std::cout << "Final test loss: " << history["test_loss"].back() << std::endl;
    std::cout << "Final test accuracy: " << history["test_accu"].back() << std::endl;
    
}

//  ./build/mnist /home/shiv/Desktop/Pytorch_project/data /home/shiv/Desktop/Pytorch_project/model/convnext.pth




