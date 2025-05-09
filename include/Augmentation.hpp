#ifndef AUGMENTATION_HPP
#define AUGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

/**
 * @brief A utility class for applying random image augmentations using OpenCV.
 * 
 * This class provides typical image augmentation techniques used in deep learning training pipelines, 
 * such as flipping, rotation, color jitter, resized cropping, and erasing.
*/
class Augmentations {
    public:
        /**
         * @brief Constructor for initializing augmentation mode.
         * @param is_training If true, applies augmentations; otherwise, leaves the image unchanged.
        */
        Augmentations(bool is_training);
    
        /**
         * @brief Applies a series of random augmentations to the input RGB image.
         * @param rgb Input image (in-place modification).
        */
        void apply(cv::Mat &rgb);

        ~Augmentations();
    
    private:
        bool is_training_; 
    
        /**
         * @brief Applies a random horizontal flip to the image.
         * @param rgb Input/output image.
        */
        void randomHorizontalFlip(cv::Mat &rgb);
    
        /**
         * @brief Rotates the image by a random angle within [-max_angle, max_angle].
         * @param rgb Input/output image.
         * @param max_angle Maximum rotation angle in degrees.
        */
        void randomRotation(cv::Mat &rgb, double max_angle);
    
        /**
         * @brief Applies color jitter transformations.
         * @param image Input/output image.
         * @param brightness Brightness factor.
         * @param contrast Contrast factor.
         * @param saturation Saturation factor.
         * @param hue Hue factor.
        */
        void randomColorJitter(cv::Mat &image, float brightness, float contrast, float saturation, float hue);
    
        /**
         * @brief Applies a random resized crop to the image.
         * @param image Input/output image.
         * @param out_height Target crop height.
         * @param out_width Target crop width.
         * @param min_scale Minimum scale ratio.
         * @param max_scale Maximum scale ratio.
        */
        void randomResizedCrop(cv::Mat &image, int out_height, int out_width, float min_scale, float max_scale);
    
        /**
         * @brief Randomly erases a portion of the image.
         * @param image Input/output image.
         * @param p Probability of applying the erasing.
        */
        void randomErasing(cv::Mat &image, float p);
};


#endif // AUGMENTATION_HPP
