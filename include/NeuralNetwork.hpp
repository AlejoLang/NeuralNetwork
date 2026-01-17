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
    Matrix<double> foward(Matrix<double> input);
    void backwards(Matrix<double> target);
    void update(double learningRate);

  public:
    NeuralNetwork(std::vector<int> layersConfig, int batchSize = 1);
    NeuralNetwork::TrainResponse train(std::vector<std::vector<double>> inputs,
                                       std::vector<std::vector<double>> outputs,
                                       float trainingUseRatio, int epochs = 1, int batchSize = 32,
                                       double learningRate = 0.01, double learningRateUpdate = 1);
};