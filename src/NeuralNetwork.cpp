#include "../include/NeuralNetwork.hpp"
#include <cmath>
#include <iostream>

template <typename T> T sigmoid(T x) {
    return (1 / (1 + pow(M_E, -x)));
}
template <typename T> T softmax(T x) {
    return (pow(M_E, -x) / pow((1 + pow(M_E, -x)), 2));
}

NeuralNetwork::NeuralNetwork(std::vector<int> layersConfig)
    : output(1, layersConfig[layersConfig.size() - 1]) {
    this->layersConfig = layersConfig;
    for (size_t i = 1; i < layersConfig.size() - 1;
         ++i) { // Creates the layers ignoring the first one since it doesnt need weights or biases
        Layer newLayer(layersConfig[i], layersConfig[i - 1], Layer::SIGMOID);
        this->layers.push_back(newLayer);
    }
    Layer lastLayer(layersConfig[layersConfig.size() - 1], layersConfig[layersConfig.size() - 2],
                    Layer::SOFTMAX);
    this->layers.push_back(lastLayer);
}

void NeuralNetwork::randomize() {
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i].initRandom();
    }
}

Matrix<double> NeuralNetwork::foward(Matrix<double> input) {
    for (size_t i = 0; i < this->layers.size(); i++) {
        input = this->layers[i].foward(input);
    }
    this->output = input;
    return input;
}

void NeuralNetwork::backwards(Matrix<double> target) {
    Matrix<double> firstStepDeltas = (output - target);
    this->layers[this->layers.size() - 1].setDeltas(
        firstStepDeltas); // Set the deltas of the output layer as the cost function
    Matrix<double> backwardsResultDeltas = firstStepDeltas;
    for (int i = this->layers.size() - 2; i >= 0; --i) {
        backwardsResultDeltas =
            this->layers[i].backwards(this->layers[i + 1].getWeights(), backwardsResultDeltas);
    }
}

void NeuralNetwork::update() {
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i].update(0.01);
    }
}