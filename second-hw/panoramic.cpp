/**
 * @author Riccardo De Zen. 2019295.
 */
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include "panoramic_utils.h"
#include "panoramic.h"

const int PanoramicImage::RIGHT = 0;
const int PanoramicImage::LEFT = 1;

PanoramicImage::PanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction) {
    // Moving vector because it is passed by value on purpose.
    this->original_images = std::move(images);
    this->half_fov = half_fov;
    this->dist_ratio = dist_ratio;

    // If images are right to left, flip their order.
    if (direction == PanoramicImage::LEFT)
        std::reverse(original_images.begin(), original_images.end());
}

cv::Mat PanoramicImage::get(bool gray, bool equalize, bool draw) {
    int gray_i = gray;
    int equal_i = equalize;
    bool should_draw = draw && match_images.empty();
    cv::Mat &result = results[gray_i][equal_i];

    // Return image if already computed, but recompute if I need to draw and I haven't before.
    if (!result.empty() && !should_draw)
        return result;

    // Recompute shifts only if I need to draw and I haven't before.
    if (shift_x.empty() || should_draw)
        prepareShifts((draw) ? &match_images : nullptr);

    // Prepare material images vector if not prepared yet.
    auto N = projected_images.size();

    // projected_images and projected_gray are both already available after feature matching.
    if (gray) {
        if (equalize && gray_equalized.empty()) {
            gray_equalized.resize(N);
            for (auto i = 0; i < N; i++)
                gray_equalized[i] = PanoramicImage::equalize(projected_gray[i]);
        }
    } else {
        if (equalize && bgr_equalized.empty()) {
            bgr_equalized.resize(N);
            for (auto i = 0; i < N; i++)
                bgr_equalized[i] = PanoramicImage::equalize(projected_images[i]);
        }
    }

    return makePanoramic(*(material_vectors[gray_i][equal_i]), result);
}

std::vector<cv::Mat> PanoramicImage::getAll(bool draw) {
    std::vector<cv::Mat> result(4);
    result[0] = get(false, false, draw);
    result[1] = get(false, true, draw);
    result[2] = get(true, false, draw);
    result[3] = get(true, true, draw);
    cv::cvtColor(result[2], result[2], cv::COLOR_GRAY2BGR);
    cv::cvtColor(result[3], result[3], cv::COLOR_GRAY2BGR);
    return result;
}

std::vector<cv::Mat> PanoramicImage::matchImages() {
    return match_images;
}

void PanoramicImage::projectImages() {
    auto N = original_images.size();
    projected_images.resize(N);
    projected_gray.resize(N);
    for (auto i = 0; i < N; i++) {
        // Project image on cylinder
        projected_images[i] = PanoramicUtils::cylindricalProj(original_images[i], half_fov);
        // Convert to grayscale for feature detection
        cv::cvtColor(projected_images[i], projected_gray[i], cv::COLOR_BGR2GRAY);
    }
}

cv::Mat PanoramicImage::equalize(const cv::Mat &image) {
    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    cv::Mat output;

    for (auto &p : planes)
        cv::equalizeHist(p, p);

    cv::merge(planes, output);
    return output;
}

void PanoramicImage::prepareShifts(std::vector<cv::Mat> *draw_destination) {
    if (projected_images.empty())
        projectImages();

    auto N = projected_images.size();

    // Pre-load keypoint and descriptor vectors.
    std::vector<std::vector<cv::KeyPoint>> key_points(N);
    std::vector<cv::Mat> descriptors(N);

    // Find key points and descriptors for each image.
    cv::Ptr<cv::Feature2D> detector = getDetector();
    for (auto i = 0; i < N; i++)
        detector->detectAndCompute(projected_gray[i], cv::noArray(), key_points[i], descriptors[i]);

    // Create matcher.
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> all_matches(N - 1);

    // Match pairs (all but last image).
    for (auto i = 0; i < N - 1; i++)
        matcher->match(descriptors[i], descriptors[i + 1], all_matches[i]);

    // Find minimum distance and take only the matches that are below such distance * dist_ratio.
    for (auto &matches : all_matches) {
        std::vector<cv::DMatch> close_matches;
        // Find minimum distance.
        auto min_distance = matches[0].distance;
        for (auto &match : matches)
            if (match.distance < min_distance)
                min_distance = match.distance;

        // Set min_distance to at least 1.
        min_distance = std::max(1.0f, min_distance);

        // Only take matches that are below threshold.
        for (auto &match : matches)
            if (match.distance <= (min_distance * dist_ratio))
                close_matches.push_back(match);

        // Replace matches.
        matches = close_matches;
    }

    // Find appropriate matches with ransac and compute average distance between pictures.
    shift_x.resize(all_matches.size());
    shift_y.resize(all_matches.size());

    // Left and right margins.
    int cumulative_x = 0;
    left_x = 0;
    right_x = 0;

    int cumulative_y = 0;
    // Max vertical margins (determine the final container image's size).
    // Used also to cut out the final image.
    upper_y = 0;
    lower_y = 0;

    for (auto i = 0; i < all_matches.size(); i++) {
        // Get points in the two images.
        std::vector<cv::Point2f> left_points;
        std::vector<cv::Point2f> right_points;
        for (auto j = 0; j < all_matches[i].size(); j++) {
            // Get the key points from the good matches
            left_points.push_back(key_points[i][all_matches[i][j].queryIdx].pt);
            right_points.push_back(key_points[i + 1][all_matches[i][j].trainIdx].pt);
        }
        std::vector<int> mask;
        std::vector<cv::DMatch> homography_matches;
        findHomography(left_points, right_points, mask, cv::RANSAC);

        double sum_dx = 0, sum_dy = 0;
        int count_dx = 0, count_dy = 0;
        for (auto j = 0; j < mask.size(); j++) {
            if (!mask[j])
                continue;
            // Copy best matches to draw them later.
            homography_matches.push_back(all_matches[i][j]);
            // Images go right, always positive.
            sum_dx += left_points[j].x - right_points[j].x;
            // Images can go up and down, average can be 0. Take lowest and highest values found.
            sum_dy += left_points[j].y - right_points[j].y;
            count_dx++;
            count_dy++;
        }

        // Replace old matches.
        all_matches[i] = homography_matches;

        // Truncating is wiser. If I round I risk a segmentation fault.
        shift_x[i] = (int) round(sum_dx / count_dx);
        shift_y[i] = (int) round(sum_dy / count_dy);

        // The total shift (allows negatives).
        cumulative_y += shift_y[i];
        cumulative_x += shift_x[i];

        // Left, right, top, bottom margins (to handle negative shifts).
        left_x = std::min(left_x, cumulative_x);
        right_x = std::max(right_x, cumulative_x);
        upper_y = std::min(upper_y, cumulative_y);
        lower_y = std::max(lower_y, cumulative_y);
    }

    // Draw the matches if requested.
    if (draw_destination != nullptr) {
        draw_destination->resize(all_matches.size());
        for (auto i = 0; i < all_matches.size(); i++) {
            cv::drawMatches(
                    projected_gray[i], key_points[i],
                    projected_gray[i + 1], key_points[i + 1],
                    all_matches[i], (*draw_destination)[i],
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
            );
        }
    }
}

cv::Mat PanoramicImage::makePanoramic(
        const std::vector<cv::Mat> &material_images,
        cv::Mat &result_dest
) {
    auto N = projected_images.size();
    int width = projected_images[0].cols;
    int height = projected_images[0].rows;
    int channels = material_images[0].channels();

    int total_height = height + lower_y - upper_y;
    int total_width = width + right_x - left_x;

    result_dest = cv::Mat(total_height, total_width, material_images[0].type());

    // Drawing position of current image.
    int curr_x = -left_x;
    int curr_y = -upper_y;

    // Other variables for smoothing the junctions.
    int curr_piece_left = 0;
    int curr_junction = 0;
    int smooth_half_span = 0;
    int smooth_start = 0;
    int smooth_end = 0;
    double smooth_span = 0;
    double alpha = 0;

    // Stitch images together.
    for (auto i = 0; i < N; i++) {
        // Vertical range is always whole projected image.
        cv::Range vert_range(curr_y, curr_y + height);

        // We want to crop the image to mitigate the distortion at the sides.
        // If the shift is of 10, we'll have the two images meet in the middle.
        // Since I paste left to right I only need to trim to the left.
        // I multiply by 0.6 instead of dividing by 2 to leave room for linear interpolation.
        curr_piece_left = (i > 0) ? (int) round((width - shift_x[i - 1]) * 0.6) : 0;
        curr_junction = (i > 0) ? (width - shift_x[i - 1]) / 2 : 0;
        smooth_half_span = curr_piece_left - curr_junction;
        smooth_start = curr_junction - smooth_half_span;
        smooth_end = curr_junction + smooth_half_span;
        smooth_span = smooth_half_span * 2;

        // Smooth. Since the left image has already been pasted, I can just average
        // the current image with the result.
        if (channels == 1) {
            // Used to avoid overflow in pixels.
            int tmp;
            for (auto r = 0; r < height; r++) {
                for (auto c = smooth_start; c < smooth_end; c++) {
                    alpha = (c - smooth_start) / smooth_span;
                    auto &old_px = result_dest.at<uchar>(curr_y + r, curr_x + c);
                    auto curr_px = material_images[i].at<uchar>(r, c);
                    tmp = (int) (((int) old_px) * (1 - alpha) + ((int) curr_px) * alpha);
                    old_px = tmp;
                }
            }
        } else {
            // Used to avoid overflow in pixels.
            cv::Vec3i tmp1, tmp2;
            for (auto r = 0; r < height; r++) {
                for (auto c = smooth_start; c < smooth_end; c++) {
                    alpha = (c - smooth_start) / smooth_span;
                    auto &old_px = result_dest.at<cv::Vec3b>(curr_y + r, curr_x + c);
                    auto curr_px = material_images[i].at<cv::Vec3b>(r, c);
                    tmp1 = old_px;
                    tmp2 = curr_px;
                    cv::addWeighted(tmp1, 1 - alpha, tmp2, alpha, 0, old_px);
                }
            }
        }

        cv::Mat piece = material_images[i](
                cv::Range(0, height),
                cv::Range(curr_piece_left, width)
        );

        // Paste image into destination.
        cv::Range hor_range(curr_x + curr_piece_left, curr_x + width);
        piece.copyTo(result_dest(vert_range, hor_range));

        if (i < N - 1) {
            curr_x += shift_x[i];
            curr_y += shift_y[i];
        }
    }

    // Crop image to remove top and bottom black borders.
    result_dest = result_dest(
            cv::Range(lower_y - upper_y, total_height + upper_y - lower_y),
            cv::Range(0, total_width)
    );

    return result_dest;
}


// SIFT ---

SIFTPanoramicImage::SIFTPanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction)
        : PanoramicImage(std::move(images), half_fov, dist_ratio, direction) {
}

cv::Ptr<cv::Feature2D> SIFTPanoramicImage::getDetector() {
    return cv::SIFT::create();
}


// ORB ---

ORBPanoramicImage::ORBPanoramicImage(std::vector<cv::Mat> images, double half_fov, double dist_ratio, int direction)
        : PanoramicImage(std::move(images), half_fov, dist_ratio, direction) {
}

cv::Ptr<cv::Feature2D> ORBPanoramicImage::getDetector() {
    return cv::ORB::create(5000);
}