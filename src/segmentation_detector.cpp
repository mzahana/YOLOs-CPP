#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <numeric>

// Include YOLO segmentation headers
#include "seg/YOLO11Seg.hpp"

bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <model_path> <image_path> <classes_path> [conf_threshold] [iou_threshold] [warmup_runs] [benchmark_runs]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  model_path     : Path to ONNX segmentation model file (.onnx)\n";
    std::cout << "  image_path     : Path to input image\n";
    std::cout << "  classes_path   : Path to class names file (.names or .txt)\n";
    std::cout << "  conf_threshold : Confidence threshold (default: 0.25)\n";
    std::cout << "  iou_threshold  : IoU threshold for NMS (default: 0.45)\n";
    std::cout << "  warmup_runs    : Number of warmup runs before timing (default: 3)\n";
    std::cout << "  benchmark_runs : Number of runs for timing average (default: 10)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << programName << " model.onnx image.jpg classes.names 0.25 0.45 5 20\n";
    std::cout << "\nNote: Pure inference timing excludes model loading, image loading,\n";
    std::cout << "      result drawing, and display operations.\n";
}

class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMs() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1000000.0; // Convert to milliseconds with decimal precision
    }
    
    double getElapsedUs() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to microseconds with decimal precision
    }
};

struct BenchmarkResults {
    std::vector<double> inference_times_ms;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double median_time_ms;
    double std_dev_ms;
    int total_objects_detected;
    
    void calculate() {
        if (inference_times_ms.empty()) return;
        
        // Sort for median calculation
        std::vector<double> sorted_times = inference_times_ms;
        std::sort(sorted_times.begin(), sorted_times.end());
        
        min_time_ms = sorted_times.front();
        max_time_ms = sorted_times.back();
        avg_time_ms = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0) / sorted_times.size();
        
        // Median
        size_t n = sorted_times.size();
        if (n % 2 == 0) {
            median_time_ms = (sorted_times[n/2 - 1] + sorted_times[n/2]) / 2.0;
        } else {
            median_time_ms = sorted_times[n/2];
        }
        
        // Standard deviation
        double variance = 0.0;
        for (double time : inference_times_ms) {
            variance += (time - avg_time_ms) * (time - avg_time_ms);
        }
        variance /= inference_times_ms.size();
        std_dev_ms = std::sqrt(variance);
    }
    
    void print() const {
        std::cout << "\n=== Pure Inference Timing Results ===" << std::endl;
        std::cout << "Benchmark runs: " << inference_times_ms.size() << std::endl;
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "Median time:  " << std::fixed << std::setprecision(3) << median_time_ms << " ms" << std::endl;
        std::cout << "Min time:     " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
        std::cout << "Max time:     " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "Std dev:      " << std::fixed << std::setprecision(3) << std_dev_ms << " ms" << std::endl;
        std::cout << "Average FPS:  " << std::fixed << std::setprecision(1) << (1000.0 / avg_time_ms) << std::endl;
        std::cout << "Peak FPS:     " << std::fixed << std::setprecision(1) << (1000.0 / min_time_ms) << std::endl;
        std::cout << "Total objects detected: " << total_objects_detected << std::endl;
        
        std::cout << "\n=== Individual Run Times (ms) ===" << std::endl;
        for (size_t i = 0; i < inference_times_ms.size(); ++i) {
            std::cout << "Run " << std::setw(2) << (i + 1) << ": " 
                      << std::fixed << std::setprecision(3) << inference_times_ms[i] << " ms" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 8) {
        printUsage(argv[0]);
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string classesPath = argv[3];
    float confThreshold = (argc > 4) ? std::stof(argv[4]) : 0.25f;
    float iouThreshold = (argc > 5) ? std::stof(argv[5]) : 0.45f;
    int warmupRuns = (argc > 6) ? std::stoi(argv[6]) : 3;
    int benchmarkRuns = (argc > 7) ? std::stoi(argv[7]) : 10;

    // Validate files exist
    if (!fileExists(modelPath)) {
        std::cerr << "Error: Model file not found: " << modelPath << std::endl;
        return -1;
    }
    if (!fileExists(imagePath)) {
        std::cerr << "Error: Image file not found: " << imagePath << std::endl;
        return -1;
    }
    if (!fileExists(classesPath)) {
        std::cerr << "Error: Classes file not found: " << classesPath << std::endl;
        return -1;
    }

    std::cout << "=== Pure Inference Timing Benchmark ===" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Image: " << imagePath << std::endl;
    std::cout << "Classes: " << classesPath << std::endl;
    std::cout << "Confidence threshold: " << confThreshold << std::endl;
    std::cout << "IoU threshold: " << iouThreshold << std::endl;
    std::cout << "Warmup runs: " << warmupRuns << std::endl;
    std::cout << "Benchmark runs: " << benchmarkRuns << std::endl;
    std::cout << "=======================================" << std::endl;

    // Load image ONCE (not timed)
    std::cout << "Loading image..." << std::endl;
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return -1;
    }
    std::cout << "Image dimensions: " << image.cols << "x" << image.rows << " pixels" << std::endl;

    // Load class names ONCE (not timed)
    std::vector<std::string> classNames;
    std::ifstream classFile(classesPath);
    std::string line;
    while (std::getline(classFile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        classNames.push_back(line);
    }
    std::cout << "Loaded " << classNames.size() << " class names" << std::endl;

    bool useGPU = true;
    std::vector<Segmentation> lastResults;

    try {
        // Initialize model ONCE (not timed for inference)
        std::cout << "\nInitializing segmentation model..." << std::endl;
        auto start_init = std::chrono::high_resolution_clock::now();
        
        YOLOv11SegDetector detector(modelPath, classesPath, useGPU);
        
        auto end_init = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
        std::cout << "Model initialization time: " << init_time.count() << " ms (not included in inference timing)" << std::endl;

        // Warmup runs (not timed)
        std::cout << "\nPerforming " << warmupRuns << " warmup runs..." << std::endl;
        for (int i = 0; i < warmupRuns; ++i) {
            std::cout << "Warmup run " << (i + 1) << "/" << warmupRuns << "\r" << std::flush;
            std::vector<Segmentation> warmup_results = detector.segment(image, confThreshold, iouThreshold);
            lastResults = warmup_results; // Keep last results for display
        }
        std::cout << "\nWarmup completed." << std::endl;

        // Benchmark runs - PURE INFERENCE TIMING ONLY
        std::cout << "\nStarting " << benchmarkRuns << " benchmark runs..." << std::endl;
        BenchmarkResults benchmark;
        benchmark.inference_times_ms.reserve(benchmarkRuns);
        
        PrecisionTimer timer;
        int totalObjects = 0;
        
        for (int i = 0; i < benchmarkRuns; ++i) {
            std::cout << "Benchmark run " << (i + 1) << "/" << benchmarkRuns << "\r" << std::flush;
            
            // START TIMING - Pure inference only
            timer.start();
            std::vector<Segmentation> results = detector.segment(image, confThreshold, iouThreshold);
            double inference_time_ms = timer.getElapsedMs();
            // END TIMING
            
            benchmark.inference_times_ms.push_back(inference_time_ms);
            totalObjects += results.size();
            
            if (i == benchmarkRuns - 1) {
                lastResults = results; // Keep last results for display
            }
        }
        std::cout << std::endl;
        
        benchmark.total_objects_detected = totalObjects;
        benchmark.calculate();
        benchmark.print();

        // Show detection results from last run
        std::cout << "\n=== Detection Results (Last Run) ===" << std::endl;
        std::cout << "Objects detected: " << lastResults.size() << std::endl;
        
        if (!lastResults.empty()) {
            std::cout << "\nDetailed results:" << std::endl;
            for (size_t i = 0; i < lastResults.size(); ++i) {
                const auto& seg = lastResults[i];
                std::string className = "Unknown";
                
                if (seg.classId >= 0 && static_cast<size_t>(seg.classId) < classNames.size()) {
                    className = classNames[seg.classId];
                }
                
                std::cout << "  Object " << (i + 1) << ": " << className 
                          << " (ID: " << seg.classId << ")"
                          << ", Conf: " << std::fixed << std::setprecision(2) << seg.conf * 100 << "%"
                          << ", Box: (" << seg.box.x << "," << seg.box.y << "," 
                          << seg.box.width << "," << seg.box.height << ")";
                
                if (!seg.mask.empty()) {
                    int maskArea = cv::countNonZero(seg.mask);
                    int totalArea = seg.mask.rows * seg.mask.cols;
                    float maskPercentage = (float)maskArea / totalArea * 100.0f;
                    std::cout << ", Mask: " << seg.mask.cols << "x" << seg.mask.rows 
                              << " (" << std::fixed << std::setprecision(1) << maskPercentage << "%)";
                }
                std::cout << std::endl;
            }
        }

        // Show result image
        char display_choice;
        std::cout << "\nDisplay result image? (y/n): ";
        std::cin >> display_choice;
        
        if (display_choice == 'y' || display_choice == 'Y') {
            std::cout << "Generating and displaying result image..." << std::endl;
            cv::Mat resultImage = image.clone();
            
            try {
                detector.drawSegmentationsAndBoxes(resultImage, lastResults);
            } catch (...) {
                // Fallback drawing
                std::cout << "Using fallback drawing method..." << std::endl;
                for (const auto& seg : lastResults) {
                    // Draw bounding box
                    cv::rectangle(resultImage, 
                        cv::Rect(seg.box.x, seg.box.y, seg.box.width, seg.box.height),
                        cv::Scalar(0, 255, 0), 2);
                    
                    // Add class label
                    std::string label = "Unknown";
                    if (seg.classId >= 0 && static_cast<size_t>(seg.classId) < classNames.size()) {
                        label = classNames[seg.classId];
                    }
                    label += " " + std::to_string(static_cast<int>(seg.conf * 100)) + "%";
                    
                    cv::putText(resultImage, label, 
                        cv::Point(seg.box.x, seg.box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }
            }
            
            // Resize window if image is too large
            cv::Mat displayImage = resultImage;
            if (resultImage.cols > 1200 || resultImage.rows > 800) {
                double scale = std::min(1200.0 / resultImage.cols, 800.0 / resultImage.rows);
                cv::resize(resultImage, displayImage, cv::Size(), scale, scale);
                std::cout << "Image resized for display (scale: " << std::fixed << std::setprecision(2) << scale << ")" << std::endl;
            }
            
            cv::namedWindow("Segmentation Results", cv::WINDOW_AUTOSIZE);
            cv::imshow("Segmentation Results", displayImage);
            std::cout << "Press any key to close the window..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        // Optionally save result image (not timed)
        char save_choice;
        std::cout << "\nSave result image? (y/n): ";
        std::cin >> save_choice;
        
        if (save_choice == 'y' || save_choice == 'Y') {
            std::cout << "Generating result image..." << std::endl;
            cv::Mat resultImage = image.clone();
            
            try {
                detector.drawSegmentationsAndBoxes(resultImage, lastResults);
            } catch (...) {
                // Fallback drawing
                for (const auto& seg : lastResults) {
                    cv::rectangle(resultImage, 
                        cv::Rect(seg.box.x, seg.box.y, seg.box.width, seg.box.height),
                        cv::Scalar(0, 255, 0), 2);
                }
            }
            
            std::string outputPath = "benchmark_result.jpg";
            cv::imwrite(outputPath, resultImage);
            std::cout << "Result saved to: " << outputPath << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        if (useGPU) {
            std::cout << "Trying with CPU..." << std::endl;
            try {
                YOLOv11SegDetector detector(modelPath, classesPath, false);
                
                // Single test run with CPU
                PrecisionTimer timer;
                timer.start();
                std::vector<Segmentation> results = detector.segment(image, confThreshold, iouThreshold);
                double cpu_time = timer.getElapsedMs();
                
                std::cout << "CPU inference time: " << std::fixed << std::setprecision(3) 
                          << cpu_time << " ms" << std::endl;
                std::cout << "Objects detected: " << results.size() << std::endl;
                
            } catch (const std::exception& e2) {
                std::cerr << "CPU inference also failed: " << e2.what() << std::endl;
                return -1;
            }
        } else {
            return -1;
        }
    }

    return 0;
}