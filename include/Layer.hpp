#pragma once
#include "Matrix.hpp"

class Layer {

  private:
    int nodeCount;
    Matrix<double> weights;
    Matrix<double> biases;

    Matrix<double> previousLayerActivations; // a
    Matrix<double> activations;              // a
    Matrix<double> preActivations;           // z

    Matrix<double> dW;
    Matrix<double> db;

  public:
    Layer(int nodeCount, int previousLayerNodes = 0);
    void initRandom();
    int getNodeCount();
    Matrix<double> getWeights();
    Matrix<double> foward(Matrix<double>& input);
    Matrix<double> backwards(Matrix<double> nextLayerWeights, Matrix<double> nextLayerDeltas);
    void update(double learning_rate);
};
