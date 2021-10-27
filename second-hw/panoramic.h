/**
 * @author Riccardo De Zen. 2019295.
 */
#ifndef LAB5_PANORAMIC_H
#define LAB5_PANORAMIC_H

#include <opencv2/core/types.hpp>

/**
 * Base abstract class for a Panoramic image.
 * Subclasses need to implement the virtual method getDetector().
 */
class PanoramicImage {

public:

    static const int RIGHT;
    static const int LEFT;

    /**
     * @param images Vector of images sorted left to right.
     * @param half_fov Half the field of view with which the images were taken.
     * @param dist_ratio Only matches below dist_ratio times the minimum distance are considered.
     */
    explicit PanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction = RIGHT);

    /**
     * @param gray If true, compute the result with the grayscale images.
     * @param equalize If true, use equalized images to compute the result.
     * @param draw If true, also draws the matches and puts them in appropriate images. Can be retrieved with
     *        getMatchImages().
     * @return The panoramic image, generated using the class-defined features. It is computed lazily the first time
     *         this method is called, and immediately returned for subsequent calls.
     */
    cv::Mat get(bool gray = false, bool equalize = false, bool draw = false);

    /**
     * Calls all 4 combinations of `get(bool, bool, bool)` and returns the results.
     * @return Vector of 4 images, in this order: bgr, equalized bgr, grayscale, equalized grayscale. Grayscale images
     *         are also converted to BGR for easier visualization.
     */
    std::vector<cv::Mat> getAll(bool draw = false);

    /**
     * @return Vector of images containing the matches found from features. Empty if getPanoramicORB() was never
     *         called with draw = true.
     */
    std::vector<cv::Mat> matchImages();

protected:
    // Params
    double half_fov;
    double dist_ratio;

    // Shifts and such for final image creation.
    // Only need to be computed once since matches are always computed on
    // grayscale non-equalized images.
    std::vector<int> shift_x;
    std::vector<int> shift_y;
    int left_x = 0;
    int right_x = 0;
    int upper_y = 0;
    int lower_y = 0;

    // The original and cylinder projected images.
    std::vector<cv::Mat> original_images;

    std::vector<cv::Mat> projected_images;
    std::vector<cv::Mat> projected_gray;

    std::vector<cv::Mat> bgr_equalized;
    std::vector<cv::Mat> gray_equalized;

    // Pointers to above vectors for deterministic access using `get(bool, bool, bool)` params.
    std::vector<cv::Mat> *material_vectors[2][2] = {{&projected_images, &bgr_equalized},
                                                    {&projected_gray,   &gray_equalized}};

    // Resulting images for the 4 parameter combinations:
    // grayscale | grayscale equalized | BGR | BGR equalized
    // Axis 0 : grayscale.
    // Axis 1 : equalization.
    cv::Mat results[2][2] = {{cv::Mat(), cv::Mat()},
                             {cv::Mat(), cv::Mat()}};

    // The images with matches drawn on them.
    // A single vector is needed because the matches are always computed on a
    // non-equalized grayscale image.
    std::vector<cv::Mat> match_images;

    /**
     * Method defining the feature detector to use.
     */
    virtual cv::Ptr<cv::Feature2D> getDetector() = 0;

    /**
     * Project the images on a cylinder and turn them to greyscale.
     */
    void projectImages();

    /**
     * @param image The image to equalize.
     * @returns The equalized grayscale or bgr image.
     */
    static cv::Mat equalize(const cv::Mat &image);

    /**
     * Compute features, matches, and shifts. Only needs to be ran once.
     */
    void prepareShifts(std::vector<cv::Mat> *draw_destination);

    /**
     * @param material_images Images to use when making the final result.
     * @param result_dest Where to store the result to avoid computing it again.
     * @return The panoramic image, generated using the features given by the detector.
     */
    cv::Mat makePanoramic(
            const std::vector<cv::Mat> &material_images,
            cv::Mat &result_dest
    );
};


// SIFT ---

class SIFTPanoramicImage : public PanoramicImage {

public:

    explicit SIFTPanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction = RIGHT);

protected:

    cv::Ptr<cv::Feature2D> getDetector() override;

};


// ORB ---

class ORBPanoramicImage : public PanoramicImage {

public:

    explicit ORBPanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction = RIGHT);

protected:

    cv::Ptr<cv::Feature2D> getDetector() override;

};

#endif
