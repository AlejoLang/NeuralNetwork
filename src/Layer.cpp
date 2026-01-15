#include "../include/Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

double sigmoid(double x) {
    return (1 / (1 + pow(M_E, -x)));
}
double sigmoid_derivative(double x) {
    return (pow(M_E, -x) / pow((1 + pow(M_E, -x)), 2));
}

Layer::Layer(int nodeCount, int previousLayerNodes) {
    this->nodeCount = nodeCount;
    this->weights = Matrix<double>(previousLayerNodes, nodeCount);
    this->biases = Matrix<double>(1, nodeCount);
}

void Layer::initRandom() {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<double> distr(-0.5, 0.5);
    for (size_t j = 0; j < this->weights.getHeight(); ++j) {
        for (size_t i = 0; i < this->weights.getWidth(); i++) {
            this->weights.setValue(i, j, distr(generator));
        }
    }
    for (size_t j = 0; j < this->biases.getHeight(); ++j) {
        for (size_t i = 0; i < this->biases.getWidth(); i++) {
            this->biases.setValue(i, j, distr(generator));
        }
    }
}

int Layer::getNodeCount() {
    return this->nodeCount;
}

Matrix<double> Layer::getWeights() {
    return this->weights;
}

Matrix<double> Layer::foward(Matrix<double>& input) {
    this->previousLayerActivations = input;
    this->preActivations = (this->weights * input) + this->biases;
    this->activations = this->preActivations.apply(sigmoid);
    return this->activations;
}

Matrix<double> Layer::backwards(Matrix<double> nextLayerWeights, Matrix<double> nextLayerDeltas) {
    Matrix<double> deltas(1, this->nodeCount);
    deltas = (nextLayerWeights.transpose() * nextLayerDeltas);
    this->db = deltas;
    this->dW = deltas * this->previousLayerActivations.transpose();

    return deltas;
}

void Layer::update(double learning_rate) {
    this->weights = this->weights - (this->dW * learning_rate);
    this->biases = this->biases - (this->db * learning_rate);
}