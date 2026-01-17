#include "../include/Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

template <typename T> Matrix<T> sigmoid(Matrix<T> vals) {
    Matrix<T> newMat(vals.getWidth(), vals.getHeight());

    for (size_t i = 0; i < vals.getWidth(); ++i) {
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            newMat.setValue(i, j, (static_cast<T>(1) / (1 + pow(M_E, -vals.getValue(i, j)))));
        }
    }
    return newMat;
}
template <typename T> Matrix<T> sigmoid_derivative(Matrix<T> vals) {
    Matrix<T> newMat(vals.getWidth(), vals.getHeight());
    for (size_t i = 0; i < vals.getWidth(); ++i) {
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            newMat.setValue(
                i, j,
                (pow(M_E, -vals.getValue(i, j)) / pow((1 + pow(M_E, -vals.getValue(i, j))), 2)));
        }
    }
    return newMat;
}
template <typename T> Matrix<T> relu(Matrix<T> vals) {
    Matrix<T> newMat(vals.getWidth(), vals.getHeight());
    for (size_t i = 0; i < vals.getWidth(); ++i) {
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            newMat.setValue(i, j, std::max(static_cast<T>(0), vals.getValue(i, j)));
        }
    }
    return newMat;
}
template <typename T> Matrix<T> relu_derivative(Matrix<T> vals) {
    Matrix<T> newMat(vals.getWidth(), vals.getHeight());
    for (size_t i = 0; i < vals.getWidth(); ++i) {
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            newMat.setValue(i, j, vals.getValue(i, j) > 0 ? 1 : 0);
        }
    }
    return newMat;
}
template <typename T> Matrix<T> softmax(Matrix<T> vals) {
    Matrix<T> newMat(vals.getWidth(), vals.getHeight());
    for (size_t i = 0; i < vals.getWidth(); ++i) {
        T maxVal = vals.getValue(i, 0);
        for (size_t j = 1; j < vals.getHeight(); ++j) {
            maxVal = std::max(maxVal, vals.getValue(i, j));
        }
        T sum = 0;
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            sum += exp(vals.getValue(i, j) - maxVal);
        }
        for (size_t j = 0; j < vals.getHeight(); ++j) {
            newMat.setValue(i, j, exp(vals.getValue(i, j) - maxVal) / sum);
        }
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
    std::uniform_real_distribution<double> distr;
    switch (this->activationFunctionType) {
    case SIGMOID: {
        // Xavier initialization: sqrt(6 / (fan_in + fan_out))
        double limit = sqrt(6.0 / (this->weights.getHeight() + this->weights.getWidth()));
        distr = std::uniform_real_distribution<double>(-limit, limit);
        break;
    }
    case RELU: {
        // Normal (Gaussian)
        double limit = sqrt(6.0 / (this->weights.getHeight()));
        distr = std::uniform_real_distribution<double>(-limit, limit);
        break;
    }
    default:
        distr = std::uniform_real_distribution<double>(-0.5, 0.5);
        break;
    }
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
    this->deltas = d;

    // Calculate db: average of deltas across batch
    Matrix<double> avgDeltas(1, this->deltas.getHeight());
    for (size_t j = 0; j < d.getHeight(); j++) {
        double sum = 0;
        for (size_t i = 0; i < d.getWidth(); i++) {
            sum += d.getValue(i, j);
        }
        avgDeltas.setValue(0, j, sum / d.getWidth());
    }
    this->db = avgDeltas;

    // Calculate dW: (A^T * deltas) / batchSize
    this->dW = d * this->previousLayerActivations.transpose();
    // Average over batch
    for (size_t i = 0; i < this->dW.getWidth(); i++) {
        for (size_t j = 0; j < this->dW.getHeight(); j++) {
            this->dW.setValue(i, j, this->dW.getValue(i, j) / d.getWidth());
        }
    }
}

Matrix<double> Layer::getWeights() {
    return this->weights;
}

Matrix<double> Layer::getBiases() {
    return this->biases;
}

Matrix<double> Layer::foward(Matrix<double>& input) {
    this->previousLayerActivations = input;
    this->preActivations = (this->weights * input);
    for (size_t i = 0; i < this->preActivations.getWidth(); ++i) {
        for (size_t j = 0; j < this->preActivations.getHeight(); ++j) {
            this->preActivations.setValue(
                i, j, this->preActivations.getValue(i, j) + this->biases.getValue(0, j));
        }
    }

    this->activations = this->activationFunction(this->preActivations);
    return this->activations;
}

Matrix<double> Layer::backwards(Matrix<double> nextLayerWeights, Matrix<double> nextLayerDeltas) {
    Matrix<double> deltas(nextLayerDeltas.getWidth(), this->nodeCount);

    if (this->activationFunctionType != SOFTMAX) {
        deltas = (nextLayerWeights.transpose() * nextLayerDeltas);
        deltas = deltas.hadamard(this->activationDerivative(this->preActivations));
    } else {
        deltas = nextLayerDeltas;
    }
    this->deltas = deltas;

    // Calculate db: average of deltas across batch
    Matrix<double> avgDeltas(1, this->deltas.getHeight());
    for (size_t j = 0; j < deltas.getHeight(); j++) {
        double sum = 0;
        for (size_t i = 0; i < deltas.getWidth(); i++) {
            sum += deltas.getValue(i, j);
        }
        avgDeltas.setValue(0, j, sum / deltas.getWidth());
    }
    this->db = avgDeltas;
    // Calculate dW: (A^T * deltas) / batchSize
    this->dW = deltas * this->previousLayerActivations.transpose();
    // Average over batch
    for (size_t i = 0; i < this->dW.getWidth(); i++) {
        for (size_t j = 0; j < this->dW.getHeight(); j++) {
            this->dW.setValue(i, j, this->dW.getValue(i, j) / deltas.getWidth());
        }
    }

    return deltas;
}

void Layer::update(double learning_rate) {
    this->weights = this->weights - (this->dW * learning_rate);
    this->biases = this->biases - (this->db * learning_rate);
}