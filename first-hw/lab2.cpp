#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;
using cv::utils::fs::glob;

static void show_usage(const string &name) {
    std::cerr << "Usage: " << name << " [options]\n"
              << "Options:\n"
              // Help option
              << "\t-h, --help\t\tShow this help message.\n"
              // Data directory option
              << "\t-d, --data DIR\t\tPath of the directory containing calibration .png images."
              << " Defaults to \"./lab2_data/checkerboard_images/\".\n"
              // Test file option
              << "\t-t, --test FILE\t\tPath of the file to then use as a test image."
              << " Defaults to \"./lab2_data/test_image.png\".\n"
              // Width and height of checkerboard pattern.
              << "\t-c, --columns WIDTH\tWidth (columns) of checkerboard pattern. Integer, defaults to 6.\n"
              << "\t-r, --rows HEIGHT\tHeight (rows) of checkerboard pattern. Integer, defaults to 5."
              << std::endl;
}

// Just a structure to keep information about a camera in one place.
struct CalibratedCamera {
    // Intrinsics matrix.
    Mat camera_matrix;
    // Distortion coefficients.
    vector<double> distortion_coefficients;
};

/**
 * @param files The set of filenames for the images.
 * @return A vector containing the images.
 */
vector<Mat> getCheckerboardImages(const vector<string> &files);

/**
 * @param checkerboard_images The images in which to find corners.
 * @param pattern_size The size of the checkerboard pattern (columns by rows).
 * @return A vector with the coordinates of the corners on the images.
 * @throws invalid_argument if an image has missing corners.
 */
vector<vector<Vec2f>> getImagePoints(const vector<Mat> &checkerboard_images, const Size &pattern_size);

/**
 * @param n How many patterns are needed.
 * @param rows How many rows each pattern has.
 * @param columns How many columns each pattern has.
 * @param unit_size The size, in meters, of each square of the checkerboard.
 * @return A Vector of object points, essentially a vector of n identical vectors, each containing the same
 *         rows * columns points.
 */
vector<vector<Vec3f>> getObjectPoints(int n, int rows, int columns, float unit_size);

/**
 * Compute the reprojection error for an image.
 *
 * @param object_points The object points in the world reference frame.
 * @param pred_image_points The points found in the image.
 * @param rot The rotation vector between world frame and camera frame.
 * @param tra The translation vector between world frame and camera frame.
 * @param camera_matrix The 3 by 3 camera matrix for intrinsic params.
 * @param dist The distortion coefficients vectors.
 * @return The reprojection error for the given object.
 */
double reprojectionError(
        const vector<Vec3f> &object_points, const vector<Vec2f> &pred_image_points,
        const Mat &rot, const Mat &tra, const Mat &camera_matrix, const vector<double> &dist
);

/**
 * Compute the mean RMS for a set of images.
 *
 * @param single_RMS The RMS for single images.
 */
double meanRMS(const vector<double> &single_RMS);

/**
 *
 * @param camera The camera information (intrinsics and distortion coefficients).
 * @param image The image to rectify.
 * @return An image obtained rectifying the input one with the parameters of the given camera.
 */
Mat undistortImage(const CalibratedCamera &camera, const Mat &image);

int main(int argc, char **argv) {
    // Default options.
    string DATA_DIR = "./lab2_data/checkerboard_images/";
    string TEST_IMG = "./lab2_data/test_image.png";
    Size PATTERN_SIZE = Size(6, 5);
    float UNIT = 0.11;

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
            } else if ((arg == "-d") || (arg == "--data")) {
                // No data directory -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the directory.
                DATA_DIR = argv[++i];
            } else if ((arg == "-t") || (arg == "--test")) {
                // No test image -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the file.
                TEST_IMG = argv[++i];
            } else if ((arg == "-c") || (arg == "--columns")) {
                // No value -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the width.
                PATTERN_SIZE = Size(stoi(argv[++i]), PATTERN_SIZE.height);
            } else if ((arg == "-r") || (arg == "--rows")) {
                // No value -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the height.
                PATTERN_SIZE = Size(PATTERN_SIZE.width, stoi(argv[++i]));
            }
        }
    }

    // Find all png files and load them.
    vector<string> checkerboard_files;
    glob(DATA_DIR, "*.png", checkerboard_files);
    vector<Mat> checkerboard_images = getCheckerboardImages(checkerboard_files);

    // Find checkerboard corners on the images.
    vector<vector<Vec2f>> image_points = getImagePoints(checkerboard_images, PATTERN_SIZE);

    // Show example of found corners.
    // Image 17 of the given dataset performs worst later, hence why I deem it an interesting example.
    // If another dataset is used, just take the last image.
    int EXAMPLE_IMAGE = (16 > checkerboard_images.size() - 1) ? checkerboard_images.size() : 16;
    Mat example_drawn = checkerboard_images[EXAMPLE_IMAGE].clone();
    drawChessboardCorners(example_drawn, PATTERN_SIZE, image_points[EXAMPLE_IMAGE], true);
    resize(example_drawn, example_drawn, Size(560, 480));
    namedWindow("Main");
    imshow("Main", example_drawn);

    // Calibrate camera.
    vector<vector<Vec3f>> object_points = getObjectPoints(
            (int) image_points.size(), PATTERN_SIZE.height, PATTERN_SIZE.width, UNIT
    );
    Size size(checkerboard_images[0].cols, checkerboard_images[0].rows);

    Mat camera_matrix;
    vector<Mat> rot;
    vector<Mat> tra;
    vector<double> dist;
    vector<double> dev_in;
    vector<double> dev_ex;
    vector<double> errors;
    double mean_error = calibrateCamera(
            object_points, image_points, size, camera_matrix,
            dist, rot, tra, dev_in, dev_ex,
            errors, 0
    );

    // Computing errors manually for assignment.
    vector<double> manual_errors;
    manual_errors.reserve(object_points.size());
    for (int i = 0; i < object_points.size(); i++) {
        manual_errors.push_back(reprojectionError(
                object_points[i], image_points[i],
                rot[i], tra[i], camera_matrix, dist
        ));
    }
    double manual_mean_error = meanRMS(manual_errors);

    // Printing mean error, parameters, and camera matrix.
    cout << "Returned mean RMS: " << mean_error << endl;
    cout << "Manual mean RMS: " << manual_mean_error << endl;
    cout << "Distortion parameters: {k1: "
         << dist[0] << ", k2: " << dist[1] << ", p1: "
         << dist[2] << ", p2: " << dist[3] << ", k3: "
         << dist[4] << "}." << endl;

    // Printing the intrinsics matrix along with intrinsics names.
    // I realize this looks contrived, but it did not seem worth developing a more generic function.
    cout << "Camera matrix:" << endl << setprecision(5) << "[ "
         << "au = " << setw(6) << camera_matrix.at<double>(0, 0) << " "
         << "     " << camera_matrix.at<double>(0, 1) << "      "
         << "uc = " << setw(6) << camera_matrix.at<double>(0, 2) << " ]\n[ "
         << "     " << camera_matrix.at<double>(1, 0) << "      "
         << "av = " << setw(6) << camera_matrix.at<double>(1, 1) << " "
         << "vc = " << setw(6) << camera_matrix.at<double>(1, 2) << " ]\n[ "
         << "     " << camera_matrix.at<double>(2, 0) << "      "
         << "     " << camera_matrix.at<double>(2, 1) << "      "
         << "     " << camera_matrix.at<double>(2, 2) << "      ]" << endl;

    // Used for debug reasons to compare single image scores.
    /*
    for (auto &e: errors)
        std::cout << e << ' ';
    cout << endl;
    for (auto &e: manual_errors)
        std::cout << e << ' ';
    */

    // Images with the minimum and maximum errors.
    int worst_image_index = (int) (min_element(manual_errors.begin(), manual_errors.end()) - manual_errors.begin());
    int best_image_index = (int) (max_element(manual_errors.begin(), manual_errors.end()) - manual_errors.begin());
    cout << "Best performing image was: " << checkerboard_files[worst_image_index]
         << " with error " << manual_errors[worst_image_index] << endl;
    cout << "Worst performing image was: " << checkerboard_files[best_image_index]
         << " with error " << manual_errors[best_image_index] << endl;

    // Rectify an image using the information of the calibrated camera.
    CalibratedCamera camera = {camera_matrix, dist};
    Mat test_image = imread(TEST_IMG);
    Mat rectified_image = undistortImage(camera, test_image);

    // Show the result.
    Mat comparison;
    hconcat(test_image, rectified_image, comparison);
    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", comparison);
    waitKey(0);

    return 0;
}

vector<Mat> getCheckerboardImages(const vector<string> &files) {
    vector<Mat> images;
    images.reserve(files.size());

    // Load images.
    for (auto &file : files)
        images.push_back(imread(file));

    return images;
}

vector<vector<Vec2f>> getImagePoints(const vector<Mat> &checkerboard_images, const Size &pattern_size) {
    vector<vector<Vec2f>> image_points;
    image_points.reserve(checkerboard_images.size());

    for (auto &img: checkerboard_images) {
        vector<Vec2f> corners;
        bool res = findChessboardCorners(img, pattern_size, corners, 0);
        // Raise error if no corners found.
        if (!res)
            throw invalid_argument("Image had missing corners.");
        // Refine to sub-pixel precision.
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        cornerSubPix(
                gray, corners, Size(15, 15), Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001)
        );
        image_points.push_back(corners);
    }

    return image_points;
}

vector<vector<Vec3f>> getObjectPoints(int n, int rows, int columns, float unit_size) {
    vector<vector<Vec3f>> obj_points;

    // Each pattern contains the same set of points, because all patterns are the same checkerboard.
    for (int i = 0; i < n; i++) {
        vector<Vec3f> patt;
        for (int j = 0; j < rows; j++)
            for (int k = 0; k < columns; k++)
                patt.emplace_back(((float) j) * unit_size, ((float) k) * unit_size, 0);
        obj_points.push_back(patt);
    }
    return obj_points;
}

double reprojectionError(
        const vector<Vec3f> &object_points, const vector<Vec2f> &pred_image_points,
        const Mat &rot, const Mat &tra, const Mat &camera_matrix, const vector<double> &dist
) {
    // Project the points and compute the distance with the predicted ones.
    vector<Vec2f> image_points;
    projectPoints(object_points, rot, tra, camera_matrix, dist, image_points);
    return sqrt(norm(pred_image_points, image_points, NORM_L2SQR) / image_points.size());
}

double meanRMS(const vector<double> &single_RMS) {
    vector<double> squared_RMS;
    pow(single_RMS, 2, squared_RMS);
    Scalar mean_RMS = mean(squared_RMS);
    return sqrt(mean_RMS[0]);
}

Mat undistortImage(const CalibratedCamera &camera, const Mat &image) {
    Size original_size = Size(image.cols, image.rows);

    // Get camera matrix for target image.
    Rect roi;
    Mat new_camera_matrix = getOptimalNewCameraMatrix(
            camera.camera_matrix, camera.distortion_coefficients, original_size, 1, original_size, &roi
    );

    // Get maps from new camera matrix.
    Mat map_x, map_y;
    initUndistortRectifyMap(
            camera.camera_matrix, camera.distortion_coefficients, Mat(),
            new_camera_matrix, original_size, CV_32FC1, map_x, map_y
    );

    // Remap and crop.
    Mat rectified;
    remap(image, rectified, map_x, map_y, INTER_LINEAR);
    rectified = rectified(roi);

    // Resize image, since it has been cropped.
    resize(rectified, rectified, original_size);

    return rectified;
}