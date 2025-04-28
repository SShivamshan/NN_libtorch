#ifndef TRAINER_TPP
#define TRAINER_TPP

// Guards for the projection of files : https://stackoverflow.com/a/54363250 
#ifndef TRAINER_HPP
#error __FILE__ should only be included from Trainer.hpp.
#endif // TRAINER_HPP

template <typename ModelType>
Trainer<ModelType>::Trainer(torch::optim::Optimizer* optimizer_, int num_epochs_, torch::Device device_,
                            bool save_model_, const boost::filesystem::path save_path_,size_t train_size, bool binary, int num_classes)
    : optimizer(optimizer_),
      num_epochs(num_epochs_),
      device(device_),
      save_model(save_model_),
      save_path(save_path_),
      train_size_(train_size),
      binary_(binary),
      num_classes(num_classes) {}


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
    auto [avg_precision, avg_recall, avg_f1] = metrics_acc.average();
    std::cout << "Avg Precision: " << avg_precision
          << " | Avg Recall: " << avg_recall
          << " | Avg F1 Score: " << avg_f1 << std::endl;

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
        torch::Tensor prediction;
        optimizer->zero_grad();

        output = model->forward({data}); 
        // std::cout << output.sizes() << std::endl;
        // std::cout << target.sizes() << std::endl;
        if(!binary_){
            loss = torch::nn::functional::cross_entropy(output, target);
            loss.backward();
            optimizer->step();

            prediction = output.argmax(1);

        }else{
            // output shape [batch_size, 1]
            target = target.view({-1, 1});  // matching target shape 
            loss = torch::nn::functional::binary_cross_entropy_with_logits(output, target);  // Binary loss
            loss.backward();
            optimizer->step();

            // For binary classification, threshold logits at 0.5 
            prediction = (output > 0.5).to(torch::kLong);
        }
        
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
    float precision, recall,f1;
    torch::NoGradGuard no_grad;
    std::vector<int> preds_vec, targets_vec;

    for (const auto& batch : test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        torch::Tensor loss;
        torch::Tensor prediction;

        auto output = model->forward({data});
        if(!binary_){
            loss = torch::nn::functional::cross_entropy(output, target);
            prediction = output.argmax(1);
        }else{
            loss = torch::nn::functional::binary_cross_entropy_with_logits(output, target);
            prediction = output > 0.5;
        }
        
        // Calculate statistics
        auto pred_cpu = prediction.to(torch::kCPU);
        auto target_cpu = target.to(torch::kCPU);

        for (int i = 0; i < pred_cpu.size(0); ++i) {
            preds_vec.push_back(pred_cpu[i].template item<int>());
            targets_vec.push_back(target_cpu[i].template item<int>());
        }
        if(binary_){
            std::tie(precision, recall, f1) = compute_batch_metrics_binary(targets_vec, preds_vec);
        }else{
            std::tie(precision,recall,f1) = compute_batch_metrics_multiclass(targets_vec,preds_vec,num_classes);
        }
        metrics_acc.add(precision, recall, f1);
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