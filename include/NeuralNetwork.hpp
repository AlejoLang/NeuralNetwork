#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include <vector>

class NeuralNetwork {
  private:
    std::vector<int> layersConfig;
    std::vector<Layer> layers;
    Matrix<double> output;

  public:
    NeuralNetwork(std::vector<int> layersConfig, int batchSize = 1);
    void randomize();
    Matrix<double> foward(Matrix<double> input);
    void backwards(Matrix<double> target);
    void update();
};