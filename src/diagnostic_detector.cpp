#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <iomanip>

// Include YOLO segmentation headers
#include "seg/YOLO11Seg.hpp"

bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <model_path> <image_path> <classes_path> [conf_threshold] [iou_threshold]\n";
    std::cout << "\nThis diagnostic version helps debug detection issues.\n";
    std::cout << "It shows detailed information about overlapping detections and NMS filtering.\n";
}

// Calculate IoU between two bounding boxes
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    int intersection = (x2 - x1) * (y2 - y1);
    int union_area = box1.area() + box2.area() - intersection;
    
    return static_cast<float>(intersection) / union_area;
}

// Calculate distance between box centers
float calculateCenterDistance(const cv::Rect& box1, const cv::Rect& box2) {
    int cx1 = box1.x + box1.width / 2;
    int cy1 = box1.y + box1.height / 2;
    int cx2 = box2.x + box2.width / 2;
    int cy2 = box2.y + box2.height / 2;
    
    return std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
}

void analyzeDetections(const std::vector<Segmentation>& results, const std::vector<std::string>& classNames) {
    std::cout << "\n=== Detection Analysis ===" << std::endl;
    
    if (results.size() <= 1) {
        std::cout << "Only " << results.size() << " detection(s) found. No overlap analysis needed." << std::endl;
        return;
    }
    
    std::cout << "Found " << results.size() << " detections. Analyzing overlaps..." << std::endl;
    
    // Analyze each pair of detections
    for (size_t i = 0; i < results.size(); ++i) {
        for (size_t j = i + 1; j < results.size(); ++j) {
            const auto& det1 = results[i];
            const auto& det2 = results[j];
            
            cv::Rect box1(det1.box.x, det1.box.y, det1.box.width, det1.box.height);
            cv::Rect box2(det2.box.x, det2.box.y, det2.box.width, det2.box.height);
            
            float iou = calculateIoU(box1, box2);
            float distance = calculateCenterDistance(box1, box2);
            
            std::string class1 = (det1.classId >= 0 && static_cast<size_t>(det1.classId) < classNames.size()) 
                               ? classNames[det1.classId] : "Unknown";
            std::string class2 = (det2.classId >= 0 && static_cast<size_t>(det2.classId) < classNames.size()) 
                               ? classNames[det2.classId] : "Unknown";
            
            std::cout << "\nDetection pair " << (i+1) << " vs " << (j+1) << ":" << std::endl;
            std::cout << "  Det " << (i+1) << ": " << class1 << " (" << std::fixed << std::setprecision(1) 
                      << det1.conf * 100 << "%) at (" << det1.box.x << "," << det1.box.y 
                      << "," << det1.box.width << "," << det1.box.height << ")" << std::endl;
            std::cout << "  Det " << (j+1) << ": " << class2 << " (" << std::fixed << std::setprecision(1) 
                      << det2.conf * 100 << "%) at (" << det2.box.x << "," << det2.box.y 
                      << "," << det2.box.width << "," << det2.box.height << ")" << std::endl;
            std::cout << "  IoU: " << std::fixed << std::setprecision(3) << iou << std::endl;
            std::cout << "  Center distance: " << std::fixed << std::setprecision(1) << distance << " pixels" << std::endl;
            
            if (iou > 0.1) {
                std::cout << "  ⚠️  HIGH OVERLAP detected! IoU = " << iou << std::endl;
                if (det1.classId == det2.classId) {
                    std::cout << "  ⚠️  SAME CLASS - this should have been filtered by NMS!" << std::endl;
                }
            }
            
            if (distance < 50) {
                std::cout << "  ⚠️  CLOSE PROXIMITY - centers only " << distance << " pixels apart" << std::endl;
            }
        }
    }
    
    // Suggestions
    std::cout << "\n=== Recommendations ===" << std::endl;
    if (results.size() > 1) {
        // Check if detections are same class
        bool sameClass = true;
        int firstClass = results[0].classId;
        for (const auto& det : results) {
            if (det.classId != firstClass) {
                sameClass = false;
                break;
            }
        }
        
        if (sameClass) {
            std::cout << "• All detections are the same class - try LOWER IoU threshold (e.g., 0.2-0.3)" << std::endl;
            std::cout << "• Or try HIGHER confidence threshold to eliminate weak detections" << std::endl;
        } else {
            std::cout << "• Detections are different classes - this might be correct" << std::endl;
        }
        
        // Check confidence difference
        float maxConf = 0, minConf = 1;
        for (const auto& det : results) {
            maxConf = std::max(maxConf, det.conf);
            minConf = std::min(minConf, det.conf);
        }
        
        if (maxConf - minConf > 0.2) {
            std::cout << "• Large confidence difference (" << std::fixed << std::setprecision(1) 
                      << (maxConf - minConf) * 100 << "%) - consider higher confidence threshold" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        printUsage(argv[0]);
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string classesPath = argv[3];
    float confThreshold = (argc > 4) ? std::stof(argv[4]) : 0.25f;
    float iouThreshold = (argc > 5) ? std::stof(argv[5]) : 0.45f;

    // Validate files exist
    if (!fileExists(modelPath) || !fileExists(imagePath) || !fileExists(classesPath)) {
        std::cerr << "Error: One or more files not found!" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        return -1;
    }

    // Load class names
    std::vector<std::string> classNames;
    std::ifstream classFile(classesPath);
    std::string line;
    while (std::getline(classFile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        classNames.push_back(line);
    }

    std::cout << "=== Diagnostic Segmentation Analysis ===" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Image: " << imagePath << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    std::cout << "Confidence threshold: " << confThreshold << std::endl;
    std::cout << "IoU threshold: " << iouThreshold << std::endl;
    std::cout << "=======================================" << std::endl;

    try {
        YOLOv11SegDetector detector(modelPath, classesPath, true);
        
        std::cout << "Running inference..." << std::endl;
        std::vector<Segmentation> results = detector.segment(image, confThreshold, iouThreshold);
        
        std::cout << "Detected " << results.size() << " objects." << std::endl;
        
        // Detailed analysis
        analyzeDetections(results, classNames);
        
        // Draw and display results
        if (!results.empty()) {
            cv::Mat resultImage = image.clone();
            
            // Draw each detection with different colors
            std::vector<cv::Scalar> colors = {
                cv::Scalar(0, 255, 0),    // Green
                cv::Scalar(255, 0, 0),    // Blue  
                cv::Scalar(0, 0, 255),    // Red
                cv::Scalar(255, 255, 0),  // Cyan
                cv::Scalar(255, 0, 255),  // Magenta
                cv::Scalar(0, 255, 255)   // Yellow
            };
            
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& seg = results[i];
                cv::Scalar color = colors[i % colors.size()];
                
                // Draw bounding box
                cv::rectangle(resultImage, 
                    cv::Rect(seg.box.x, seg.box.y, seg.box.width, seg.box.height),
                    color, 3);
                
                // Add label with detection number
                std::string label = "Det" + std::to_string(i + 1) + ": ";
                if (seg.classId >= 0 && static_cast<size_t>(seg.classId) < classNames.size()) {
                    label += classNames[seg.classId];
                }
                label += " " + std::to_string(static_cast<int>(seg.conf * 100)) + "%";
                
                cv::putText(resultImage, label, 
                    cv::Point(seg.box.x, seg.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
                
                // Draw center point
                int cx = seg.box.x + seg.box.width / 2;
                int cy = seg.box.y + seg.box.height / 2;
                cv::circle(resultImage, cv::Point(cx, cy), 5, color, -1);
            }
            
            // Resize for display if needed
            cv::Mat displayImage = resultImage;
            if (resultImage.cols > 1200 || resultImage.rows > 800) {
                double scale = std::min(1200.0 / resultImage.cols, 800.0 / resultImage.rows);
                cv::resize(resultImage, displayImage, cv::Size(), scale, scale);
            }
            
            cv::namedWindow("Diagnostic Results", cv::WINDOW_AUTOSIZE);
            cv::imshow("Diagnostic Results", displayImage);
            std::cout << "\nPress any key to close..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
            
            // Save diagnostic image
            cv::imwrite("diagnostic_result.jpg", resultImage);
            std::cout << "Diagnostic image saved as: diagnostic_result.jpg" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}