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
        void randomColorJitter(cv::Mat &image, float brightness, float contrast, float saturation, float hue);
        void randomResizedCrop(cv::Mat &image, int out_height, int out_width, float min_scale, float max_scale);
        void randomErasing(cv::Mat &image, float p);
};


#endif // AUGMENTATION_HPP