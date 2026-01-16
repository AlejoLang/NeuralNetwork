#pragma once
#include "Matrix.hpp"

class Layer {
  public:
    enum ActivationFunction { SIGMOID, RELU, SOFTMAX };

  private:
    int nodeCount;
    Matrix<double> weights;
    Matrix<double> biases;
    ActivationFunction activationFunctionType;
    Matrix<double> (*activationFunction)(Matrix<double>);
    Matrix<double> (*activationDerivative)(Matrix<double>);

    Matrix<double> previousLayerActivations;
    Matrix<double> activations;    // a
    Matrix<double> preActivations; // z

    Matrix<double> deltas;
    Matrix<double> dW;
    Matrix<double> db;

  public:
    Layer(int nodeCount, int previousLayerNodes = 0, ActivationFunction activationF = SIGMOID);
    void initRandom();
    int getNodeCount();
    void setDeltas(Matrix<double> d);
    Matrix<double> getWeights();
    Matrix<double> foward(Matrix<double>& input);
    Matrix<double> backwards(Matrix<double> nextLayerWeights, Matrix<double> nextLayerDeltas);
    void update(double learning_rate);
};
