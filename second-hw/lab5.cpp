/**
 * @author Riccardo De Zen. 2019295.
 */
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "panoramic_utils.h"
#include "panoramic.h"

using namespace std;
using namespace cv;
using cv::utils::fs::glob;

static void show_usage(const string &name) {
    std::cerr << "Usage: " << name << " [options]\n"
              << "Options:\n"
              // Help option
              << "\t-h, --help\t\tShow this help message.\n"
              // Data directory option
              << "\t-p, --path DIR\t\tPath of the directory containing the images to use."
              << " Defaults to \"./lab5_data/lab/\".\n"
              // Image format option
              << "\t-s, --suffix SUFFIX\tImage extension. Defaults to \"bmp\".\n"
              // Field of view option
              << "\t-f, --fov ANGLE\t\tField of view of the camera used to take the pictures."
              << " Defaults to 66.\n"
              // Direction of pictures
              << "\t-d, --direction l|r\tDirection of the picture. \"l\" for right to left, \"r\" for left to right."
              << " Defaults to \"r\"."
              << std::endl;
}

/**
 * @param files Image files to get.
 * @return A vector containing the images.
 */
vector<Mat> getImages(const vector<string> &files);

int main(int argc, char **argv) {
    // Default options.
    string DATA_DIR = "./lab5_data/lab/";
    string SUFFIX = "bmp";
    double FOV = 66;
    int DIRECTION = PanoramicImage::RIGHT;

    // Command line arguments parsing ---
    if (argc > 1) {
        // Gave an option and no value.
        if (argc < 3) {
            show_usage(argv[0]);
            return 1;
        }
        for (int i = 0; i < argc; i++) {
            string arg = argv[i];

            if ((arg == "-h") || (arg == "--help")) {
                show_usage(argv[0]);
                return 0;
            } else if ((arg == "-p") || (arg == "--path")) {
                // No data directory -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the directory.
                DATA_DIR = argv[++i];
            } else if ((arg == "-s") || (arg == "--suffix")) {
                // No test image -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the file.
                SUFFIX = argv[++i];
            } else if ((arg == "-f") || (arg == "--fov")) {
                // No value -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the angle.
                FOV = stod(argv[++i]);
            } else if ((arg == "-d") || (arg == "--direction")) {
                // No value -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Set direction if value is appropriate.
                string val = argv[++i];
                if (val == "r")
                    DIRECTION = PanoramicImage::RIGHT;
                else if (val == "l")
                    DIRECTION = PanoramicImage::LEFT;
                else {
                    show_usage(argv[0]);
                    return 1;
                }
            }
        }
    }

    // Load images and make Panoramic image.
    vector<string> image_files;
    glob(DATA_DIR, "*." + SUFFIX, image_files);
    vector<Mat> images = getImages(image_files);

    // Linear interpolation is enabled by default. I did not think it should have been a separate option.
    // It is found at lines 257 - 284 of panoramic.cpp
    // SIFT with 10 distance ratio already works on all datasets.
    SIFTPanoramicImage sift_image(images, FOV / 2, 10, DIRECTION);
    vector<Mat> sift_results = sift_image.getAll(true);
    Mat sift_comparison;
    cv::vconcat(sift_results, sift_comparison);
    namedWindow("SIFT", WINDOW_NORMAL);
    imshow("SIFT", sift_comparison);

    namedWindow("SIFT match example", WINDOW_NORMAL);
    imshow("SIFT match example", sift_image.matchImages()[0]);

    waitKey();

    return 0;
}

vector<Mat> getImages(const vector<string> &files) {
    vector<Mat> images;
    images.reserve(files.size());

    // Load images.
    for (auto &file : files)
        images.push_back(imread(file));

    return images;
}

