#include "Augmentation.hpp"

Augmentations::Augmentations(bool is_training): is_training_(is_training){}
Augmentations::~Augmentations(){}

// Solution : https://www.programiz.com/cpp-programming/lambda-expression 
void Augmentations::apply(cv::Mat &rgb) {
    if (is_training_) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Each entry is a pair of:
        // 1. A lambda function to perform an augmentation with a capture clause from reference ([&] all reference will be captured)
        // 2. A probability of applying that augmentation
        std::vector<std::pair<std::function<void()>, double>> augmentations;
        augmentations.push_back({[&]() { randomHorizontalFlip(rgb ); }, 0.5});
        augmentations.push_back({[&]() { randomRotation(rgb, 15); }, 0.3});
        augmentations.push_back({[&]() { centerCrop(rgb, 224, 224); }, 0.4});
        if (!rgb.empty()) {
            augmentations.push_back({[&]() { randomBrightnessContrast(rgb); }, 0.4});
        }

        std::shuffle(augmentations.begin(), augmentations.end(), gen);

        for (const auto& aug : augmentations) {
            if (dis(gen) < aug.second) {
                aug.first();
            }
        }
    } else {
        // Test-time augmentation: Only center cropping for consistency
        centerCrop(rgb, 224, 224);
    }
}
void Augmentations::randomHorizontalFlip(cv::Mat &rgb) {
    // std::cout << "Apply horizontal flip" << std::endl;
    if (rand() % 2) { // 50% chance to flip
        if (!rgb.empty()) cv::flip(rgb, rgb, 1);
    }
}

void Augmentations::randomRotation(cv::Mat &rgb, double max_angle) {
    // std::cout << "Random rotation applying " << std::endl;
    double angle = ((rand() % (int)(2 * max_angle)) - max_angle);
    cv::Point2f center(rgb.cols / 2.0, rgb.rows / 2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

    if (!rgb.empty()) cv::warpAffine(rgb, rgb, rotation_matrix, rgb.size());
}

void Augmentations::centerCrop(cv::Mat &rgb, int crop_w, int crop_h) {
    // std::cout << "Applying center crop of size " << crop_w << "x" << crop_h << std::endl;
    // std::cout << "image size : " << rgb.cols << ", " << rgb.rows << std::endl;
    // Validate input dimensions
    int img_width =  rgb.cols;
    int img_height = rgb.rows;
    if (crop_w > img_width || crop_h > img_height) {
        std::cerr << "Error: Crop size exceeds image dimensions!" << std::endl;
        return;
    }
    
    // Compute crop position (centered)
    int x = (img_width - crop_w) / 2;
    int y = (img_height - crop_h) / 2;
    cv::Rect crop_roi(x, y, crop_w, crop_h);

    if (!rgb.empty()) {
        cv::Mat padded_rgb = cv::Mat::zeros(rgb.size(), rgb.type());
        cv::Mat center_crop_rgb = rgb(crop_roi);
        center_crop_rgb.copyTo(padded_rgb(crop_roi));
        rgb = padded_rgb;
    }
}

void Augmentations::randomBrightnessContrast(cv::Mat &rgb) {
    // std::cout << "Applying random brightness and contrast" << std::endl;

    // Random brightness and contrast adjustment
    double alpha = 0.8 + (rand() % 5) / 10.0; // Ensures range [0.8, 1.2]
    if (alpha < 0.1) alpha = 0.1; // Prevents division issues

    int beta = (rand() % 41) - 20; // Ensures range [-20, 20]

    try {
        // Apply contrast and brightness, ensuring valid output
        rgb.convertTo(rgb, -1, alpha, beta);

        // Clip values between 0 and 255
        cv::threshold(rgb, rgb, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(rgb, rgb, 0, 0, cv::THRESH_TOZERO);
    } catch (cv::Exception &e) {
        std::cerr << "OpenCV Exception in randomBrightnessContrast: " << e.what() << std::endl;
    }
}

