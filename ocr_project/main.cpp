#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>

int main() {
    // Load image
    cv::Mat img = cv::imread("/home/wena/Desktop/ocr_project/Cars1.png");

    // Check if image is loaded successfully
    if (img.empty()) {
        std::cout << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Reduce noise with bilateral filter
    cv::Mat filtered_gray;
    cv::bilateralFilter(gray, filtered_gray, 11, 17, 17);

    // Edge detection using Canny
    cv::Mat edged;
    cv::Canny(filtered_gray, edged, 30, 200);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    // Find contour that is rectangular (assuming license plate)
    std::vector<cv::Point> license_plate;
    for (auto& contour : contours) {
        double peri = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.018 * peri, true);

        if (approx.size() == 4) {
            license_plate = approx;
            break;
        }
    }

    // Crop and process the license plate region
    if (!license_plate.empty()) {
        cv::Rect plate_rect = cv::boundingRect(license_plate);
        cv::Mat plate_img = gray(plate_rect);

        // Initialize Tesseract
        tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
        if (ocr->Init(NULL, "eng")) {
            std::cout << "Could not initialize Tesseract." << std::endl;
            return -1;
        }

        // OCR on the cropped license plate image
        ocr->SetImage(plate_img.data, plate_img.cols, plate_img.rows, 1, plate_img.step);
        ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);

        // Extract text from license plate
        char* text = ocr->GetUTF8Text();
        std::cout << "Detected License Plate Text: " << text << std::endl;

        // Cleanup
        ocr->End();
        delete[] text;

        // Display the license plate region
        cv::imshow("License Plate", plate_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } else {
        std::cout << "License plate not found." << std::endl;
    }

    return 0;
}
