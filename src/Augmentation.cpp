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
        augmentations.push_back({[&]() { randomRotation(rgb, 15); }, 0.4});
        augmentations.push_back({[&]() { randomResizedCrop(rgb, rgb.rows, rgb.cols, 0.8, 1.0); }, 0.4});
        augmentations.push_back({[&]() { randomColorJitter(rgb, 0.2, 0.2, 0.2, 0.1); }, 0.5});
        augmentations.push_back({[&]() { randomErasing(rgb, 0.2); }, 0.3});
        
        std::shuffle(augmentations.begin(), augmentations.end(), gen);

        for (const auto& aug : augmentations) {
            if (dis(gen) < aug.second) {
                aug.first();
            }
        }
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

void Augmentations::randomResizedCrop(cv::Mat &image, int out_height, int out_width, float min_scale, float max_scale) {
    // Get random scale and aspect ratio
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> scale_dist(min_scale, max_scale);
    std::uniform_real_distribution<> aspect_dist(0.75, 1.33);
    
    float random_scale = scale_dist(gen);
    float random_aspect = aspect_dist(gen);
    
    int img_height = image.rows;
    int img_width = image.cols;
    
    int crop_height = static_cast<int>(img_height * random_scale);
    int crop_width = static_cast<int>(crop_height * random_aspect);
    
    // crop dimensions so they don't exceed image dimensions
    crop_height = std::min(crop_height, img_height);
    crop_width = std::min(crop_width, img_width);
    
    // Get random crop position
    std::uniform_int_distribution<> x_dist(0, img_width - crop_width);
    std::uniform_int_distribution<> y_dist(0, img_height - crop_height);
    
    int x = x_dist(gen);
    int y = y_dist(gen);
    
    // Crop and resize
    cv::Rect crop_rect(x, y, crop_width, crop_height);
    cv::Mat cropped = image(crop_rect);
    cv::resize(cropped, image, cv::Size(out_width, out_height), 0, 0, cv::INTER_LINEAR);
}

void Augmentations::randomColorJitter(cv::Mat &image, float brightness, float contrast, float saturation, float hue) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Convert to HSV for easier manipulation
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    // Brightness adjustment in V channel
    if (brightness > 0) {
        std::uniform_real_distribution<> bright_dist(-brightness, brightness);
        float delta = bright_dist(gen) * 255;
        
        for (int i = 0; i < hsv.rows; i++) {
            for (int j = 0; j < hsv.cols; j++) {
                hsv.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(i, j)[2] + delta);
            }
        }
    }
    
    // Saturation adjustment in S channel
    if (saturation > 0) {
        std::uniform_real_distribution<> sat_dist(std::max(0.0f, 1 - saturation), 1 + saturation);
        float factor = sat_dist(gen);
        
        for (int i = 0; i < hsv.rows; i++) {
            for (int j = 0; j < hsv.cols; j++) {
                hsv.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(i, j)[1] * factor);
            }
        }
    }
    
    // Hue adjustment in H channel
    if (hue > 0) {
        std::uniform_real_distribution<> hue_dist(-hue * 180, hue * 180);
        int delta = static_cast<int>(hue_dist(gen));
        
        for (int i = 0; i < hsv.rows; i++) {
            for (int j = 0; j < hsv.cols; j++) {
                int h = hsv.at<cv::Vec3b>(i, j)[0] + delta;
                if (h < 0) h += 180;
                if (h >= 180) h -= 180;
                hsv.at<cv::Vec3b>(i, j)[0] = h;
            }
        }
    }
    
    // Convert back to BGR
    cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
    
    // Contrast adjustment in BGR
    if (contrast > 0) {
        std::uniform_real_distribution<> con_dist(std::max(0.0f, 1 - contrast), 1 + contrast);
        float factor = con_dist(gen);
        
        image.convertTo(image, -1, factor, 0);
    }
}

void Augmentations::randomErasing(cv::Mat &image, float p) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) > p) return;
    
    // Random erasing parameters
    std::uniform_real_distribution<> scale_dist(0.02, 0.2);
    std::uniform_real_distribution<> aspect_dist(0.3, 3.3);
    std::uniform_int_distribution<> color_dist(0, 255);
    
    float area_scale = scale_dist(gen);
    float aspect_ratio = aspect_dist(gen);
    
    int img_h = image.rows;
    int img_w = image.cols;
    
    int erase_area = static_cast<int>(area_scale * img_h * img_w);
    int erase_h = static_cast<int>(sqrt(erase_area * aspect_ratio));
    int erase_w = static_cast<int>(sqrt(erase_area / aspect_ratio));
    
    // Make sure dimensions are valid
    erase_h = std::min(erase_h, img_h);
    erase_w = std::min(erase_w, img_w);
    
    // Random position
    std::uniform_int_distribution<> x_dist(0, img_w - erase_w);
    std::uniform_int_distribution<> y_dist(0, img_h - erase_h);
    
    int x = x_dist(gen);
    int y = y_dist(gen);
    
    // Fill with random color or mean value
    cv::Rect rect(x, y, erase_w, erase_h);
    
    // Create random color
    cv::Vec3b color;
    for (int c = 0; c < 3; c++) {
        color[c] = static_cast<uchar>(color_dist(gen));
    }
    
    cv::rectangle(image, rect, cv::Scalar(color[0], color[1], color[2]), -1);
}

