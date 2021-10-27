#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * Base class for a square convolutional filter.
 */
class Filter {

public:

    /**
     * @param filter_size Size of the filter in pixels.
     */
    explicit Filter(int filter_size);

    /**
     * Apply filter to an image.
     * @param image The image to filter.
     * @param output The output of the convolution.
     */
    virtual void apply(cv::Mat &image, cv::Mat &output);

    /**
     * Change the filter size.
     * @param size The new size.
     */
    void setSize(int size);

    /**
     * Get the filter size.
     * @return The filter's size.
     */
    int getSize();

// Fields.

protected:

    int filter_size;

};

/**
 * Class for a Gaussian filter.
 */
class GaussianFilter : public Filter {

public:
    /**
     * @param filter_size Size of the filter.
     * @param sigma The standard deviation for the filter.
     */
    GaussianFilter(int filter_size, double sigma);

    void setSigma(double sigma);

    void apply(cv::Mat &image, cv::Mat &output) override;

protected:

    // Standard deviation (sigma) for the Filter.
    double sigma;

};

/**
 * Class for a Median Filter.
 */
class MedianFilter : public Filter {

public:

    /**
     * @param filter_size Size of the filter in pixels.
     */
    explicit MedianFilter(int filter_size);

    void apply(cv::Mat &image, cv::Mat &output) override;

};

/**
 * Class for Bilateral Filter.
 */
class BilateralFilter : public Filter {

public:

    /**
     * @param filter_size Size of the filter in pixels.
     */
    explicit BilateralFilter(int filter_size, double sigma_range, double sigma_space);

    void apply(cv::Mat &image, cv::Mat &output) override;

    void setSigmaRange(double sigma);

    void setSigmaSpace(double sigma);


protected:

    // Color range.
    double sigma_range;

    // Space.
    double sigma_space;

};