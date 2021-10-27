#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace std;
using namespace cv;

static void show_usage(const string &name) {
    std::cerr << "Usage: " << name << " [options]\n"
              << "Options:\n"
              // Help option
              << "\t-h, --help\t\tShow this help message.\n"
              // Target image option
              << "\t-i, --image FILE\tPath to the image to use."
              << " Defaults to \"./lab3_data/data/image.jpg\"."
              << std::endl;
}

struct Userdata {
    Mat &current_image;
    Filter &filter;
};

// Apparently in C++ constants are a bad practice according to CLion. This seems to be a suitable workaround to make
// window names available globally.

string GAUSS_WIN() { return "Gaussian Filter"; }

string MEDIAN_WIN() { return "Median Filter"; }

string BILATERAL_WIN() { return "Bilateral Filter"; }

/**
 * Compare three images horizontally in a single window.
 * Window will be either 1280 pixels wide or the concatenated images' width, whichever is smallest.
 * @returns The name of the window.
 */
string tripleComparison(
        const Mat &image1, const string &name1,
        const Mat &image2, const string &name2,
        const Mat &image3, const string &name3
);

/**
 * Compute histograms of an image.
 * @param image Target image, assumed to be in BGR space, three channels, 256 color levels each.
 */
vector<Mat> getHistogramsBGR(const Mat &image);

/**
 * Equalize an image while showing its histograms before and after.
 * @param image The image to equalize. Assumed to have 256 color levels per channel.
 * @param output The equalized image. All three channels will be equalized.
 */
void equalizeAndShowBGR(Mat &image, Mat &output);

/**
 * Equalize an image on the HSV space. Only equalizes V channel. Shows before and after histograms on BGR space.
 * @param image The image to equalize. Must be BGR, will be converted inside the function.
 * @param output The equalized image. Only V channel will be equalized. Will be reconverted to BGR.
 */
void equalizeAndShowHSV(Mat &image, Mat &output);

/**
 * Show a set of three histograms.
 * @param hists The histograms.
 * @param window_suffix The suffix for the windows' names.
 */
void showHistogram(std::vector<cv::Mat> &hists, const std::string &window_suffix);

/**
 * Detects a click on an image and puts its x coordinate in the given variable.
 *
 * @param event The mouse event.
 * @param x The x position of the mouse.
 * @param y The y position of the mouse.
 * @param flags The flags (unused).
 * @param userdata Pointer to the int variable containing the x coordinate of the click event.
 */
void selectImage(int event, int x, int y, int flags, void *userdata);

/**
 * Update the image with the new filter size.
 * @param param The value of the trackbar. Size of the filter.
 * @param userdata The GaussianFilter.
 */
void gauss_size_update(int param, void *userdata);

/**
 *  Update the image with the new filter sigma.
 * @param param The value of the trackbar. Sigma of the filter.
 * @param userdata The GaussianFilter.
 */
void gauss_sigma_update(int param, void *userdata);

/**
 * Update image with the new filter size.
 * @param param The value of the trackbar. Size of the filter.
 * @param userdata The MedianFilter.
 */
void median_size_update(int param, void *userdata);


/**
 * Update image with the new filter color range.
 * @param param The value of the trackbar. Color range of the filter.
 * @param userdata The BilateralFilter.
 */
void bilateral_range_update(int param, void *userdata);

/**
 * Update image with the new filter space.
 * @param param The value of the trackbar. Space size of the filter.
 * @param userdata The BilateralFilter.
 */
void bilateral_space_update(int param, void *userdata);


int main(int argc, char **argv) {
    // Load image and equalize it.
    string IMAGE_PATH = "./lab3_data/data/image.jpg";

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
            } else if ((arg == "-i") || (arg == "--image")) {
                // No data directory -> error.
                if (argv[i + 1] == nullptr) {
                    show_usage(argv[0]);
                    return 1;
                }
                // Skip next argument cause it is the directory.
                IMAGE_PATH = argv[++i];
            }
        }
    }

    // Equalize and compare results.
    Mat images[] = {imread(IMAGE_PATH), Mat(), Mat()}; // Original - BGR - HSV
    equalizeAndShowBGR(images[0], images[1]);
    equalizeAndShowHSV(images[0], images[2]);
    string comparison_window = tripleComparison(
            images[0], "Original", images[1], "Equalized (BGR)", images[2], "Equalized (HSV)"
    );
    waitKey(1);

    // Busy waiting until user clicks on an image.
    int x = -1;
    setMouseCallback(comparison_window, selectImage, &x);
    while (x < 0) waitKey(1);

    // Compute which image was clicked based on x coordinate.
    Mat playground_image = images[x * 3 / getWindowImageRect(comparison_window).width];

    // Window to observe Gaussian Filter.
    string gauss_win = GAUSS_WIN();
    GaussianFilter gauss_filter(5, 1);
    namedWindow(gauss_win, WINDOW_NORMAL);
    struct Userdata gauss_userdata = {playground_image, gauss_filter};
    createTrackbar("gauss_sigma", gauss_win, nullptr, 40, gauss_sigma_update, (void *) &gauss_userdata);
    createTrackbar("gauss_size", gauss_win, nullptr, 40, gauss_size_update, (void *) &gauss_userdata);
    imshow(gauss_win, playground_image);

    // Window to observe Median Filter.
    string median_win = MEDIAN_WIN();
    MedianFilter median_filter(5);
    namedWindow(median_win, WINDOW_NORMAL);
    struct Userdata median_userdata = {playground_image, median_filter};
    createTrackbar("median_size", median_win, nullptr, 40, median_size_update, (void *) &median_userdata);
    imshow(median_win, playground_image);

    // Window to observe Bilateral Filter.
    string bilateral_win = BILATERAL_WIN();
    BilateralFilter bilateral_filter(5, 1, 1);
    namedWindow(bilateral_win, WINDOW_NORMAL);
    struct Userdata bilateral_userdata = {playground_image, bilateral_filter};
    createTrackbar("bilateral_range", bilateral_win, nullptr, 40, bilateral_range_update, (void *) &bilateral_userdata);
    createTrackbar("bilateral_space", bilateral_win, nullptr, 40, bilateral_space_update, (void *) &bilateral_userdata);
    imshow(bilateral_win, playground_image);

    waitKey(0);
    return 0;
}

string tripleComparison(
        const Mat &image1, const string &name1,
        const Mat &image2, const string &name2,
        const Mat &image3, const string &name3
) {
    string sep = " | ";
    string comparison_window = name1 + sep + name2 + sep + name3;

    // Concatenate three images.
    Mat comparison;
    hconcat(image1, image2, comparison);
    hconcat(comparison, image3, comparison);

    // Resize to fit the screen.
    int new_width = (1280 > comparison.cols) ? comparison.cols : 1280;
    int new_height = new_width * comparison.rows / comparison.cols;
    resize(comparison, comparison, Size(new_width, new_height));
    namedWindow(comparison_window);
    imshow(comparison_window, comparison);

    return comparison_window;
}

vector<Mat> getHistogramsBGR(const Mat &image) {
    // Split the image into the BGR planes.
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    // Compute the histograms.
    int num_bins = 256;
    int channels[] = {0};
    float range[] = {0, 256};
    const float *range_p = {range};
    bool uniform = true, accumulate = false;

    vector<Mat> histograms(3);
    calcHist(&bgr_planes[0], 1, channels, Mat(), histograms[0], 1, &num_bins, &range_p, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, channels, Mat(), histograms[1], 1, &num_bins, &range_p, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, channels, Mat(), histograms[2], 1, &num_bins, &range_p, uniform, accumulate);

    return histograms;
}

void equalizeAndShowBGR(Mat &image, Mat &output) {
    // Split the image into the BGR planes.
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    // Get original histograms.
    vector<Mat> histograms = getHistogramsBGR(image);

    // Equalize the channels and merge them back into a single image.
    vector<Mat> eq_bgr_planes(3);
    equalizeHist(bgr_planes[0], eq_bgr_planes[0]);
    equalizeHist(bgr_planes[1], eq_bgr_planes[1]);
    equalizeHist(bgr_planes[2], eq_bgr_planes[2]);
    merge(eq_bgr_planes, output);

    // Compute new histograms.
    vector<Mat> eq_histograms = getHistogramsBGR(output);

    // Show histograms.
    showHistogram(histograms, " - Original");
    showHistogram(eq_histograms, " - Equalized (BGR)");
}

void equalizeAndShowHSV(Mat &image, Mat &output) {
    // Split the image into the HSV planes.
    Mat hsv_image;
    vector<Mat> hsv_planes;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    split(hsv_image, hsv_planes);

    // Get original histograms.
    vector<Mat> histograms = getHistogramsBGR(image);

    // Equalize the value channel and merge channels back into a single image.
    Mat hsv_output;
    equalizeHist(hsv_planes[2], hsv_planes[2]);
    merge(hsv_planes, hsv_output);

    // Convert back to BGR for visualization.
    cvtColor(hsv_output, output, COLOR_HSV2BGR);
    // Compute new histograms.
    vector<Mat> eq_histograms = getHistogramsBGR(output);

    // Show histograms.
    showHistogram(histograms, " - Original");
    showHistogram(eq_histograms, " - Equalized (HSV)");
}

// Modified from the one provided to show the three histograms next to each other.
void showHistogram(std::vector<cv::Mat> &hists, const std::string &window_suffix) {

    // Min/Max computation
    double hmax[3] = {0, 0, 0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    cv::Scalar colors[3] = {
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255)
    };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++) {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++) {
            cv::line(
                    canvas[i],
                    cv::Point(j, rows),
                    cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                    hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                    1, 8, 0
            );
        }
    }
    tripleComparison(canvas[0], "blue", canvas[1], "green", canvas[2], "red" + window_suffix);
}

void selectImage(int event, int x, int y, int flags, void *userdata) {
    // Only act on left click.
    if (event == EVENT_LBUTTONDOWN) {
        int *p = (int *) userdata;
        *p = x;
    }
}

void gauss_sigma_update(int param, void *userdata) {
    auto &filter = (GaussianFilter &) ((Userdata *) userdata)->filter;
    auto &image = ((Userdata *) userdata)->current_image;
    Mat output;

    // Filter image.
    filter.setSigma(param);
    filter.apply(image, output);

    // Show image.
    imshow(GAUSS_WIN(), output);
}

void gauss_size_update(int param, void *userdata) {
    auto &filter = (GaussianFilter &) ((Userdata *) userdata)->filter;
    auto &image = ((Userdata *) userdata)->current_image;
    Mat output;

    // Filter image.
    filter.setSize(param);
    filter.apply(image, output);

    // Show image.
    imshow(GAUSS_WIN(), output);
}

void median_size_update(int param, void *userdata) {
    auto &filter = (MedianFilter &) ((Userdata *) userdata)->filter;
    auto &image = ((Userdata *) userdata)->current_image;
    Mat output;

    // Filter image.
    filter.setSize(param);
    filter.apply(image, output);

    // Show image.
    imshow(MEDIAN_WIN(), output);
}

void bilateral_range_update(int param, void *userdata) {
    auto &filter = (BilateralFilter &) ((Userdata *) userdata)->filter;
    auto &image = ((Userdata *) userdata)->current_image;
    Mat output;

    // Filter image.
    filter.setSigmaRange(param);
    filter.apply(image, output);

    // Show image.
    imshow(BILATERAL_WIN(), output);
}

void bilateral_space_update(int param, void *userdata) {
    auto &filter = (BilateralFilter &) ((Userdata *) userdata)->filter;
    auto &image = ((Userdata *) userdata)->current_image;
    Mat output;

    // Filter image.
    filter.setSigmaSpace(param);
    filter.apply(image, output);

    // Show image.
    imshow(BILATERAL_WIN(), output);
}