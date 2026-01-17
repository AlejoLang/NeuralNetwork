#include "../include/NeuralNetwork.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <matio.h>
#include <random>
#include <string>

void load_data(std::string path, std::vector<std::vector<double>>& images,
               std::vector<std::vector<double>>& labels) {
    mat_t* dataset = Mat_Open(path.c_str(), MAT_ACC_RDONLY);
    if (!dataset) {
        std::cerr << "Couldn't open the file" << std::endl;
        return;
    }
    matvar_t* dataVar = Mat_VarRead(dataset, "data");
    if (!dataVar) {
        std::cerr << "Cannot find variable 'data'\n";
        Mat_Close(dataset);
        return;
    }

    if (!dataVar->data) {
        std::cerr << "Data variable has no data\n";
        Mat_VarFree(dataVar);
        Mat_Close(dataset);
        return;
    }

    size_t rows = dataVar->dims[0]; // 784
    size_t cols = dataVar->dims[1]; // 70000

    std::cout << "Data dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Data class type: " << dataVar->class_type << std::endl;

    images = std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0));

    // Handle different data types
    if (dataVar->class_type == MAT_C_DOUBLE) {
        double* data = static_cast<double*>(dataVar->data);

        // Some MNIST .mat files store pixels as double in [0,255].
        // Detect that case and normalize to [0,1] to avoid sigmoid saturation.
        double maxPixel = 0.0;
        for (size_t idx = 0; idx < rows * cols; ++idx) {
            if (data[idx] > maxPixel) {
                maxPixel = data[idx];
            }
        }
        const double scale = (maxPixel > 1.0) ? 255.0 : 1.0;

        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                images[c][r] = data[r + c * rows] / scale;
    } else if (dataVar->class_type == MAT_C_UINT8) {
        uint8_t* data = static_cast<uint8_t*>(dataVar->data);
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                images[c][r] =
                    static_cast<double>(data[r + c * rows]) / 255.0; // Normalize to [0, 1]
    } else {
        std::cerr << "Unsupported data type: " << dataVar->class_type << std::endl;
        Mat_VarFree(dataVar);
        Mat_Close(dataset);
        return;
    }

    Mat_VarFree(dataVar);

    /* -------- Read labels -------- */
    matvar_t* labelVar = Mat_VarRead(dataset, "label");
    if (!labelVar) {
        std::cerr << "Cannot find variable 'label'\n";
        Mat_Close(dataset);
        return;
    }

    if (!labelVar->data) {
        std::cerr << "Label variable has no data\n";
        Mat_VarFree(labelVar);
        Mat_Close(dataset);
        return;
    }

    double* labels_raw = static_cast<double*>(labelVar->data);
    labels = std::vector<std::vector<double>>(cols, std::vector<double>(10, 0));

    for (size_t i = 0; i < cols; ++i) {
        labels[i][static_cast<int>(labels_raw[i])] = 1;
    }

    Mat_VarFree(labelVar);
    Mat_Close(dataset);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Dataset path missing" << std::endl;
        return 0;
    }
    std::string path = argv[1];
    if (path.find(".mat") == std::string::npos) {
        std::cerr << "Files is not a mat file" << std::endl;
        return 0;
    }
    std::vector<std::vector<double>> images;
    std::vector<std::vector<double>> labels;
    NeuralNetwork nenu = NeuralNetwork({784, 128, 10}, 64);

    load_data(path, images, labels);
    if (images.size() == 0 && labels.size() == 0) {
        std::cerr << "Error loading data" << std::endl;
        return 0;
    }
    std::cout << "Data loaded" << std::endl;

    NeuralNetwork::TrainResponse resp = nenu.train(images, labels, 0.8, 30, 64, 0.1, 1);
    std::cout << resp.averageCost << std::endl;
    std::cout << resp.maxCost << std::endl;
    std::cout << resp.minCost << std::endl;
    std::cout << resp.hitPercentage << std::endl;

    return 0;
}