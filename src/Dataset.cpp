#include "Dataset.hpp"

CustomDataset::CustomDataset(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels, Mode mode_)
              :image_paths(image_paths),labels(labels),mode_(mode_){

    this->set_augmentations();

}


torch::data::Example<> CustomDataset::get(size_t index){
    cv::Mat rgb_image,old_image;
    std::string image_path;
    int16_t label;
    int width{270},height{270};

    image_path = (*image_paths)[index]; // retrieve image path
    label = static_cast<int16_t>((*labels)[index]);

    // to tensor for labels and read image 
    old_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (old_image.empty()) {
        throw std::runtime_error("Failed to load image at " + image_path);
    }
    cv::resize(old_image, rgb_image, cv::Size(width, height));
    
    augmentations_->apply(rgb_image); 
    torch::Tensor rgb_tensor = CVtoTensor(rgb_image);

    torch::Tensor label_tensor = torch::tensor(label, torch::kInt64); 

    return {rgb_tensor, label_tensor};
}


torch::optional<size_t> CustomDataset::size() const{
    return image_paths->size();
}


void CustomDataset::set_augmentations() {
    std::cout << (mode_ == Mode::Train ? "Training images..." : "Testing images...") << std::endl;
    augmentations_ = std::make_shared<Augmentations>(mode_ == mode_);
}