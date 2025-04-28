#ifndef AUGMENTATION_HPP
#define AUGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>


class Augmentations {
    public:
        Augmentations(bool is_training);
        void apply(cv::Mat &rgb);
        ~Augmentations();
    private:
        bool is_training_;

        void randomHorizontalFlip(cv::Mat &rgb);
        void randomRotation(cv::Mat &rgb, double max_angle);
        void centerCrop(cv::Mat &rgb, int crop_w, int crop_h);
        void randomBrightnessContrast(cv::Mat &rgb);
};


#endif // AUGMENTATION_HPP