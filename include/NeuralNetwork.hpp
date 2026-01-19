#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include <vector>

class NeuralNetwork {
  public:
    struct TrainResponse {
        double averageCost;
        double minCost;
        double maxCost;
        double hitPercentage;
    };
    struct Sample {
        Matrix<double> input;
        Matrix<double> output;
    };

  private:
    std::vector<int> layersConfig;
    std::vector<Layer> layers;
    Matrix<double> output;
    void randomize();
    void backwards(Matrix<double> target);
    void update(double learningRate);

  public:
    NeuralNetwork();
    NeuralNetwork(std::vector<int> layersConfig);
    NeuralNetwork::TrainResponse train(std::vector<std::vector<double>> inputs,
                                       std::vector<std::vector<double>> outputs,
                                       float trainingUseRatio, int epochs = 1, int batchSize = 32,
                                       double learningRate = 0.01, double learningRateUpdate = 1);
    Matrix<double> foward(Matrix<double> input);
    void setLayersConfig(std::vector<int> layersConfig);
    void setLayerWeights(size_t layerIt, Matrix<double> weights);
    void setLayerBiases(size_t layerIt, Matrix<double> biases);
    void saveWeights(std::string path);
    void loadWeights(std::string path);
};