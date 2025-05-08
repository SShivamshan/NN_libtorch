#include "Dataset.hpp"

CustomDataset::CustomDataset(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels, Mode mode_)
              :image_paths(image_paths),labels(labels),mode_(mode_){

    this->set_augmentations();

}


torch::data::Example<> CustomDataset::get(size_t index) {
    cv::Mat bgr_image, rgb_image;
    std::string image_path;
    int16_t label;
    int width{224}, height{224};
    
    image_path = (*image_paths)[index]; 
    label = static_cast<int16_t>((*labels)[index]);
    
    // Load image in BGR format 
    bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
        throw std::runtime_error("Failed to load image at " + image_path);
    }
    
    // Resize the BGR image
    cv::Mat resized_bgr;
    cv::resize(bgr_image, resized_bgr, cv::Size(width, height));
    
    // Convert BGR to RGB
    cv::cvtColor(resized_bgr, rgb_image, cv::COLOR_BGR2RGB);
    if (mode_ == Mode::Train) {
        augmentations_->apply(rgb_image);
    }
    
    torch::Tensor rgb_tensor = CVtoTensor(rgb_image);
    torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);
    
    return {rgb_tensor, label_tensor};
}

torch::optional<size_t> CustomDataset::size() const{
    return image_paths->size();
}


void CustomDataset::set_augmentations() {
    std::cout << (mode_ == Mode::Train ? "Training images..." : "Testing images...") << std::endl;
    augmentations_ = std::make_shared<Augmentations>(mode_ == Mode::Train);
}