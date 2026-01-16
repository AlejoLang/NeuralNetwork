#include "../include/Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

template <typename T> Matrix<T> sigmoid(Matrix<T> vals) {
    Matrix<T> newMat(1, vals.getHeight());
    for (int i = 0; i < vals.getHeight(); ++i) {
        newMat.setValue(0, i, (static_cast<T>(1) / (1 + pow(M_E, -vals.getValue(0, i)))));
    }
    return newMat;
}
template <typename T> Matrix<T> sigmoid_derivative(Matrix<T> vals) {
    Matrix<T> newMat(1, vals.getHeight());
    for (int i = 0; i < vals.getHeight(); ++i) {
        newMat.setValue(
            0, i, (pow(M_E, -vals.getValue(0, i)) / pow((1 + pow(M_E, -vals.getValue(0, i))), 2)));
    }
    return newMat;
}
template <typename T> Matrix<T> relu(Matrix<T> vals) {
    Matrix<T> newMat(1, vals.getHeight());
    for (int i = 0; i < vals.getHeight(); ++i) {
        newMat.setValue(0, i, std::max(static_cast<T>(0), vals.getValue(0, i)));
    }
    return newMat;
}
template <typename T> Matrix<T> relu_derivative(Matrix<T> vals) {
    Matrix<T> newMat(1, vals.getHeight());
    for (int i = 0; i < vals.getHeight(); ++i) {
        newMat.setValue(0, i, vals.getValue(0, i) >= 0 ? 1 : 0);
    }
    return newMat;
}
template <typename T> Matrix<T> softmax(Matrix<T> vals) {
    Matrix<T> newMat(1, vals.getHeight());
    T maxVal = vals.getValue(0, 0);
    for (int i = 1; i < vals.getHeight(); ++i) {
        maxVal = std::max(maxVal, vals.getValue(0, i));
    }
    T sum = 0;
    for (int i = 0; i < vals.getHeight(); ++i) {
        sum += exp(vals.getValue(0, i) - maxVal);
    }
    for (int i = 0; i < vals.getHeight(); ++i) {
        newMat.setValue(0, i, exp(vals.getValue(0, i) - maxVal) / sum);
    }
    return newMat;
}

Layer::Layer(int nodeCount, int previousLayerNodes, ActivationFunction activationF) {
    this->nodeCount = nodeCount;
    this->weights = Matrix<double>(previousLayerNodes, nodeCount);
    this->biases = Matrix<double>(1, nodeCount);
    this->activationFunctionType = activationF;
    switch (activationF) {
    case SIGMOID:
        this->activationFunction = sigmoid<double>;
        this->activationDerivative = sigmoid_derivative<double>;
        break;
    case RELU:
        this->activationFunction = relu<double>;
        this->activationDerivative = relu_derivative<double>;
        break;
    case SOFTMAX:
        this->activationFunction = softmax<double>;
        this->activationDerivative = nullptr;
        break;

    default:
        break;
    }
}

void Layer::initRandom() {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    // Xavier initialization: sqrt(6 / (fan_in + fan_out))
    double limit = sqrt(6.0 / (this->weights.getHeight() + this->weights.getWidth()));
    std::uniform_real_distribution<double> distr(-limit, limit);
    for (size_t j = 0; j < this->weights.getHeight(); ++j) {
        for (size_t i = 0; i < this->weights.getWidth(); i++) {
            this->weights.setValue(i, j, distr(generator));
        }
    }
    for (size_t j = 0; j < this->biases.getHeight(); ++j) {
        for (size_t i = 0; i < this->biases.getWidth(); i++) {
            this->biases.setValue(i, j, 0.0);
        }
    }
}

int Layer::getNodeCount() {
    return this->nodeCount;
}

void Layer::setDeltas(Matrix<double> d) {
    if (d.getWidth() != 1) {
        return;
    }
    this->deltas = d;
    this->db = d;
    this->dW = d * this->previousLayerActivations.transpose();
}

Matrix<double> Layer::getWeights() {
    return this->weights;
}

Matrix<double> Layer::foward(Matrix<double>& input) {
    this->previousLayerActivations = input;
    this->preActivations = (this->weights * input) + this->biases;
    this->activations = this->activationFunction(this->preActivations);
    return this->activations;
}

Matrix<double> Layer::backwards(Matrix<double> nextLayerWeights, Matrix<double> nextLayerDeltas) {
    Matrix<double> deltas(1, this->nodeCount);

    if (this->activationFunctionType != SOFTMAX) {
        deltas = (nextLayerWeights.transpose() * nextLayerDeltas);
        deltas = deltas.hadamard(this->activationDerivative(this->preActivations));
    } else {
        deltas = nextLayerDeltas;
    }
    this->deltas = deltas;
    this->db = deltas;
    this->dW = deltas * this->previousLayerActivations.transpose();

    return deltas;
}

void Layer::update(double learning_rate) {
    this->weights = this->weights - (this->dW * learning_rate);
    this->biases = this->biases - (this->db * learning_rate);
}