#include "../include/NeuralNetwork.hpp"
#include <cmath>
#include <iostream>

double times2(double x) {
    return x * 2;
}

NeuralNetwork::NeuralNetwork(std::vector<int> layersConfig)
    : output(1, layersConfig[layersConfig.size() - 1]) {
    this->layersConfig = layersConfig;
    for (size_t i = 1; i < layersConfig.size();
         ++i) { // Creates the layers ignoring the first one since it doesnt need weights or biases
        Layer newLayer(layersConfig[i], layersConfig[i - 1]);
        this->layers.push_back(newLayer);
    }
}

void NeuralNetwork::randomize() {
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i].initRandom();
    }
}

Matrix<double> NeuralNetwork::foward(Matrix<double> input) {
    for (Layer& layer : this->layers) {
        input = layer.foward(input);
    }
    this->output = input;
    return input;
}

void NeuralNetwork::backwards(Matrix<double> target) {
    Matrix<double> firstStepDeltas = (target - output) * 2;
    Matrix<double> firstStepWeights(this->layers[this->layers.size() - 1].getNodeCount(),
                                    this->layers[this->layers.size() - 1].getNodeCount());
    Matrix<double> backwardsResultDeltas =
        this->layers[this->layers.size() - 1].backwards(firstStepWeights, firstStepDeltas);
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