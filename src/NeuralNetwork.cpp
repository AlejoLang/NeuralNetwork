#include "../include/NeuralNetwork.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

std::vector<NeuralNetwork::Sample> create_sample_vector(std::vector<std::vector<double>> input,
                                                        std::vector<std::vector<double>> output) {
    std::vector<NeuralNetwork::Sample> result;
    for (size_t i = 0; i < input.size(); i++) {
        Matrix<double> newInputMat(1, input[i].size());
        for (size_t j = 0; j < input[i].size(); j++) {
            newInputMat.setValue(0, j, input[i][j]);
        }
        Matrix<double> newOutputMat(1, output[i].size(), 0.0);
        for (size_t j = 0; j < output[i].size(); j++) {
            newOutputMat.setValue(0, j, output[i][j]);
        }
        result.push_back({newInputMat, newOutputMat});
    }
    return result;
}

NeuralNetwork::NeuralNetwork() {
    this->layersConfig = {};
    this->layers = {};
}

NeuralNetwork::NeuralNetwork(std::vector<int> layersConfig) {
    this->layersConfig = layersConfig;
    for (size_t i = 1; i < layersConfig.size() - 1;
         ++i) { // Creates the layers ignoring the first one since it doesnt need weights or biases
        Layer newLayer(layersConfig[i], layersConfig[i - 1], Layer::RELU);
        this->layers.push_back(newLayer);
    }
    Layer lastLayer(layersConfig[layersConfig.size() - 1], layersConfig[layersConfig.size() - 2],
                    Layer::SOFTMAX);
    this->layers.push_back(lastLayer);
}

void NeuralNetwork::setLayersConfig(std::vector<int> layersConfig) {
    this->layers.clear();
    this->layersConfig = layersConfig;
    for (size_t i = 1; i < layersConfig.size() - 1;
         ++i) { // Creates the layers ignoring the first one since it doesnt need weights or biases
        Layer newLayer(layersConfig[i], layersConfig[i - 1], Layer::RELU);
        this->layers.push_back(newLayer);
    }
    Layer lastLayer(layersConfig[layersConfig.size() - 1], layersConfig[layersConfig.size() - 2],
                    Layer::SOFTMAX);
    this->layers.push_back(lastLayer);
}

void NeuralNetwork::setLayerWeights(size_t layerIt, Matrix<double> weights) {
    this->layers[layerIt].setWeights(weights);
}

void NeuralNetwork::setLayerBiases(size_t layerIt, Matrix<double> biases) {
    this->layers[layerIt].setBiases(biases);
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

void NeuralNetwork::update(double learningRate) {
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i].update(learningRate);
    }
}

NeuralNetwork::TrainResponse NeuralNetwork::train(std::vector<std::vector<double>> inputs,
                                                  std::vector<std::vector<double>> outputs,
                                                  float trainingUseRatio, int epochs, int batchSize,
                                                  double learningRate, double learningRateUpdate) {
    if (inputs[0].size() != this->layersConfig[0]) {
        throw std::invalid_argument("Input sample size doesn't match the input layer size");
    }
    if (outputs[0].size() != this->layersConfig[this->layersConfig.size() - 1]) {
        throw std::invalid_argument("Output sample size doesn't match the output layer size");
    }
    if (inputs.size() != outputs.size()) {
        throw std::invalid_argument("Input and output vector sizes must match");
    }
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());

    std::vector<NeuralNetwork::Sample> samples = create_sample_vector(inputs, outputs);
    int split_index = inputs.size() * trainingUseRatio;

    std::shuffle(samples.begin(), samples.end(), generator);
    std::vector<Sample> training_data(samples.begin(), samples.begin() + split_index);
    std::vector<Sample> testing_data(samples.begin() + split_index, samples.end());
    samples.clear();

    this->randomize();

    for (size_t epochs_it = 0; epochs_it < epochs; ++epochs_it) {
        std::cout << "Epotch " << epochs_it + 1 << std::endl;
        if (epochs_it % 10 != 0 && epochs_it != 0) {
            learningRate *= learningRateUpdate;
        }
        std::shuffle(training_data.begin(), training_data.end(), generator);
        Matrix<double> batch_input(batchSize, training_data[0].input.getHeight());
        Matrix<double> batch_output(batchSize, training_data[0].output.getHeight());
        for (size_t training_vector_it = 0; (training_vector_it + batchSize) < training_data.size();
             training_vector_it += batchSize) {
            for (size_t batch_it = 0; batch_it < batchSize; ++batch_it) {
                for (size_t input_sample_it = 0; // Copies the input batch into a matrix of
                                                 // batchSize cols and input height rows
                     input_sample_it <
                     training_data[training_vector_it + batch_it].input.getHeight();
                     ++input_sample_it) {
                    batch_input.setValue(
                        batch_it, input_sample_it,
                        training_data[training_vector_it + batch_it].input.getValue(
                            0, input_sample_it));
                }
                for (size_t output_sample_it = 0; // Copies the output batch into a matrix of
                                                  // batchSize cols and output height rows
                     output_sample_it <
                     training_data[training_vector_it + batch_it].output.getHeight();
                     ++output_sample_it) {
                    batch_output.setValue(
                        batch_it, output_sample_it,
                        training_data[training_vector_it + batch_it].output.getValue(
                            0, output_sample_it));
                }
            }
            this->foward(batch_input);
            this->backwards(batch_output);
            this->update(learningRate);
        }
    }
    double cost = 0, maxCost = -MAXFLOAT, minCost = MAXFLOAT;
    int tsamples = 0;
    int hits = 0;
    for (size_t test_vector_it = 0; test_vector_it < testing_data.size(); ++test_vector_it) {
        Matrix<double> output = this->foward(testing_data[test_vector_it].input);
        double auxCost = 0;
        int posMax = 0;
        double maxVal = 0;
        for (int j = 0; j < output.getHeight(); ++j) {
            auxCost +=
                pow(testing_data[test_vector_it].output.getValue(0, j) - output.getValue(0, j), 2);
            if (auxCost > maxCost) {
                maxCost = auxCost;
            }
            if (auxCost < minCost) {
                minCost = auxCost;
            }
            if (output.getValue(0, j) > maxVal) {
                maxVal = output.getValue(0, j);
                posMax = j;
            }
        }
        if (testing_data[test_vector_it].output.getValue(0, posMax) == 1) {
            hits++;
        }
        cost += (auxCost / output.getHeight());
        tsamples++;
    }
    return NeuralNetwork::TrainResponse((cost / tsamples), maxCost, minCost,
                                        ((double)hits / tsamples) * 100);
}

void NeuralNetwork::saveWeights(std::string path) {
    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Couldnt create file" << std::endl;
        return;
    }
    size_t layersNum = this->layersConfig.size();
    file.write(reinterpret_cast<const char*>(&layersNum), sizeof(layersNum));
    file.write(reinterpret_cast<const char*>(this->layersConfig.data()), layersNum * sizeof(int));
    for (size_t i = 0; i < this->layers.size(); ++i) {
        std::vector<double> weights = this->layers[i].getWeights().getValuesVector();
        std::vector<double> biases = this->layers[i].getBiases().getValuesVector();
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
    }
    file.close();
}

void NeuralNetwork::loadWeights(std::string path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Couldnt open file" << std::endl;
        return;
    }
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (size <= 0) {
        std::cerr << "Error reading network config" << std::endl;
        return;
    }
    std::vector<int> config(size, 0);
    std::cout << size << std::endl;
    file.read(reinterpret_cast<char*>(config.data()), size * sizeof(int));
    for (int i : config) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    this->setLayersConfig(config);
    for (size_t i = 1; i < config.size(); ++i) {
        std::vector<double> weightsVec(config[i] * config[i - 1], 0);
        file.read(reinterpret_cast<char*>(weightsVec.data()),
                  (config[i] * config[i - 1]) * sizeof(double));
        Matrix layerWeights(config[i - 1], config[i], weightsVec);
        this->setLayerWeights(i - 1, layerWeights);
        std::vector<double> biasesVec(config[i]);
        file.read(reinterpret_cast<char*>(biasesVec.data()), config[i] * sizeof(double));
        Matrix layerBiases(1, config[i], biasesVec);
        this->setLayerBiases(i - 1, layerBiases);
    }
    file.close();
}