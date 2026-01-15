#include "../include/NeuralNetwork.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <matio.h>
#include <random>
#include <string>

void load_data(std::string path, std::vector<std::vector<double>>& images,
               std::vector<int>& labels) {
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
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                images[c][r] = data[r + c * rows];
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
    labels = std::vector<int>(cols, 0);

    for (size_t i = 0; i < cols; ++i)
        labels[i] = static_cast<int>(labels_raw[i]);

    Mat_VarFree(labelVar);
    Mat_Close(dataset);

    /* -------- Test output -------- */
    std::cout << "Images: " << images.size() << std::endl;
    std::cout << "Pixels per image: " << images[0].size() << std::endl;
    std::cout << "First label: " << labels[0] << std::endl;
}

struct Sample {
    Matrix<double> input;
    Matrix<double> output;
};

std::vector<Sample> create_sample_vector(std::vector<std::vector<double>> input,
                                         std::vector<int> output) {
    std::vector<Sample> result;
    for (size_t i = 0; i < input.size(); i++) {
        Matrix<double> newInputMat(1, input[i].size());
        for (size_t j = 0; j < input[i].size(); j++) {
            newInputMat.setValue(0, j, input[i][j]);
        }
        Matrix<double> newOutputMat(1, 10, 0.0);
        newOutputMat.setValue(0, output[i], 1);
        result.push_back({newInputMat, newOutputMat});
    }
    return result;
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
    std::vector<int> labels;
    NeuralNetwork nenu = NeuralNetwork({784, 200, 100, 10});

    load_data(path, images, labels);
    if (images.size() == 0 && labels.size() == 0) {
        std::cerr << "Error loading data" << std::endl;
        return 0;
    }
    std::cout << "Data loaded" << std::endl;

    std::vector<Sample> samples_vector = create_sample_vector(images, labels);
    images.clear();
    labels.clear();
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::shuffle(samples_vector.begin(), samples_vector.end(), generator);
    std::cout << "Input and output vectors created and shuffled" << std::endl;

    nenu.randomize();
    std::cout << "Weights/ biases randomized" << std::endl;

    std::cout << "Training..." << std::endl;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < (samples_vector.size() * 0.8); ++i) {
        nenu.foward(samples_vector[i].input);
        nenu.backwards(samples_vector[i].output);
        nenu.update();
    }
    std::cout << "Training finished after "
              << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                  start)
              << std::endl;

    std::cout << "Testing..." << std::endl;
    double cost = 0, maxCost = -MAXFLOAT, minCost = MAXFLOAT;
    int tsamples = 0;
    for (int i = (samples_vector.size() * 0.8); i < samples_vector.size(); ++i) {
        Matrix<double> output = nenu.foward(samples_vector[i].input);
        double auxCost = 0;
        for (int j = 0; j < output.getHeight(); ++j) {
            auxCost += pow((samples_vector[i].output.getValue(0, j) - output.getValue(0, j)), 2);
            if (auxCost > maxCost) {
                maxCost = auxCost;
            }
            if (auxCost < minCost) {
                minCost = auxCost;
            }
        }
        cost += (auxCost / output.getHeight());
        tsamples++;
    }
    std::cout << "AVG Cost: " << cost / tsamples << std::endl;
    std::cout << "Max Cost: " << maxCost << std::endl;
    std::cout << "Min Cost: " << minCost << std::endl;

    int a;
    std::cin >> a;

    return 0;
}