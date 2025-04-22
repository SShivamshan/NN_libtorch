#ifndef TRAINER_TPP
#define TRAINER_TPP

// Guards for the projection of files : https://stackoverflow.com/a/54363250 
#ifndef TRAINER_HPP
#error __FILE__ should only be included from Trainer.hpp.
#endif // TRAINER_HPP

template <typename ModelType>
Trainer<ModelType>::Trainer(torch::optim::Optimizer* optimizer_, int num_epochs_, torch::Device device_,
                            bool save_model_, const boost::filesystem::path save_path_,size_t train_size)
    : optimizer(optimizer_),
      num_epochs(num_epochs_),
      device(device_),
      save_model(save_model_),
      save_path(save_path_),
      train_size_(train_size) {}


template <typename ModelType>
Trainer<ModelType>::~Trainer() = default;
      
template <typename ModelType>
void Trainer<ModelType>::save(ModelType& model) {
    try {
        torch::save(model, save_path.string()); // Saves the full module
        std::cout << "Model saved to: " << save_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
}

template <typename ModelType>
template <typename TrainLoader, typename TestLoader>
std::map<std::string, std::vector<float>> Trainer<ModelType>::fit(ModelType& model, TrainLoader& train_loader, TestLoader& test_loader) {
    int seed = 42;
    torch::manual_seed(seed);
    float train_loss, train_accuracy, test_loss, test_accuracy;
    std::map<std::string, std::vector<float>> history;

    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "\t\tStart training" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::tie(train_loss, train_accuracy) = train(model, train_loader, epoch);
        history["train_loss"].push_back(train_loss);
        history["train_accu"].push_back(train_accuracy);

        // Evaluate on test set
        std::tie(test_loss, test_accuracy) = test(model, test_loader);
        history["test_loss"].push_back(test_loss);
        history["test_accu"].push_back(test_accuracy);

        std::cout << "----------------------------------------------" << std::endl;
    }

    if (save_model) {
        std::cout << "Saving model............................." << std::endl;
        save(model);
    }

    return history;
}

template <typename ModelType>
template <typename Dataloader>
std::tuple<float, float> Trainer<ModelType>::train(ModelType& model, Dataloader& train_loader, int epoch) {
    model->train();
    size_t total_num_correct = 0;
    float total_loss = 0.0;
    size_t num_samples = 0;
    int32_t batch_idx = 0;

    for (auto& batch : train_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        torch::Tensor output;
        torch::Tensor loss;

        optimizer->zero_grad();

        // Forward pass using the specific model's forward function
        output = model->forward({data}); 
        loss = torch::nn::functional::cross_entropy(output, target);

        loss.backward();
        optimizer->step();

        auto prediction = output.argmax(1);
        size_t num_correct = prediction.eq(target).sum().item().toInt();
        total_num_correct += num_correct;
        total_loss += loss.item().toFloat() * data.size(0);
        num_samples += data.size(0);

        if (batch_idx % 5 == 0) {
            std::printf("\rTrain Epoch [%d/%d]: [%5ld/%5ld] Loss: %.4f", epoch + 1, num_epochs, num_samples, train_size_, loss.item<float>());
            std::cout << std::flush;
        }

        batch_idx++;
    }

    float avg_loss = total_loss / num_samples;
    float accuracy = static_cast<float>(total_num_correct) / num_samples * 100.0f;

    std::cout << "\nEpoch: " << epoch + 1 << " | Training Loss: " << avg_loss 
              << " | Training Accuracy: " << accuracy << "%" << std::endl;

    return std::make_tuple(avg_loss, accuracy);
}

template <typename ModelType>
template <typename Dataloader>
std::tuple<float, float> Trainer<ModelType>::test(ModelType& model, Dataloader& test_loader) {
    model->eval();
    size_t total_num_correct = 0;
    float total_loss = 0.0;
    size_t num_samples = 0;

    torch::NoGradGuard no_grad;

    for (const auto& batch : test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        // Forward pass using the specific model's forward function
        auto output = model->forward({data});
        auto loss = torch::nn::functional::cross_entropy(output, target);

        // Calculate statistics
        auto prediction = output.argmax(1);
        size_t num_correct = prediction.eq(target).sum().item().toInt();
        total_num_correct += num_correct;
        total_loss += loss.item().toFloat() * data.size(0);
        num_samples += data.size(0);
    }

    float avg_loss = total_loss / num_samples;
    float accuracy = static_cast<float>(total_num_correct) / num_samples * 100.0f;

    std::cout << "Test Loss: " << avg_loss << " | Test Accuracy: " << accuracy << "%" << std::endl;

    return std::make_tuple(avg_loss, accuracy);
}


#endif // TRAINER_TPP