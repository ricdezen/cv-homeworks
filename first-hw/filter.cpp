#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"


/**
 * @param size Size will be force to be odd.
 */
Filter::Filter(int size) {
    if (size % 2 == 0)
        size++;
    filter_size = size;
}

// for base class do nothing (in derived classes it performs the corresponding filter)
void Filter::apply(cv::Mat &image, cv::Mat &output) {
    output = image.clone();
}

/**
 * @param size Set filter size. Will be forced to odd.
 */
void Filter::setSize(int size) {
    if (size % 2 == 0)
        size++;
    filter_size = size;
}

//get filter size
int Filter::getSize() {
    return filter_size;
}


// Write your code to implement the Gaussian, median and bilateral filters

// Gaussian Filter ---
GaussianFilter::GaussianFilter(int filter_size, double sigma) : Filter(filter_size) {
    this->sigma = sigma;
}

void GaussianFilter::setSigma(double new_sigma) {
    this->sigma = new_sigma;
}

void GaussianFilter::apply(cv::Mat &image, cv::Mat &output) {
    cv::GaussianBlur(image, output, cv::Size(filter_size, filter_size), sigma, sigma);
}


// Median Filter ---
MedianFilter::MedianFilter(int filter_size) : Filter(filter_size) {}

void MedianFilter::apply(cv::Mat &image, cv::Mat &output) {
    cv::medianBlur(image, output, filter_size);
}


// Bilateral Filter ---
BilateralFilter::BilateralFilter(int filter_size, double sigma_range, double sigma_space) : Filter(filter_size) {
    this->sigma_range = sigma_range;
    this->sigma_space = sigma_space;
    //this->filter_size = (int) (6 * sigma_space);
    this->filter_size = 15;
}

void BilateralFilter::apply(cv::Mat &image, cv::Mat &output) {
    cv::bilateralFilter(image, output, filter_size, sigma_range, sigma_space);
}

void BilateralFilter::setSigmaRange(double sigma) {
    sigma_range = sigma;
}

void BilateralFilter::setSigmaSpace(double sigma) {
    sigma_space = sigma;
    //filter_size = (int) (6 * sigma_space);
    filter_size = 15;
}