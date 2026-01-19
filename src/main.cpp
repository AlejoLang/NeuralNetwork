#include "../include/Canvas.hpp"
#include "../include/NeuralNetwork.hpp"
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_video.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <matio.h>
#include <random>
#include <string>

void load_data(std::string path, std::vector<std::vector<double>>& images,
               std::vector<std::vector<double>>& labels) {
    mat_t* dataset = Mat_Open(path.c_str(), MAT_ACC_RDONLY);
    if (!dataset) {
        std::cerr << "Couldn't open the file" << std::endl;
        return;
    }
    matvar_t* dataVar = Mat_VarRead(dataset, "data");
    if (!dataVar) {
        std::cerr << "Cannot find variable 'data'\n";
        Mat_Close(dataset);
        return;
    }

    if (!dataVar->data) {
        std::cerr << "Data variable has no data\n";
        Mat_VarFree(dataVar);
        Mat_Close(dataset);
        return;
    }

    size_t rows = dataVar->dims[0]; // 784
    size_t cols = dataVar->dims[1]; // 70000

    std::cout << "Data dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Data class type: " << dataVar->class_type << std::endl;

    images = std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0));

    // Handle different data types
    if (dataVar->class_type == MAT_C_DOUBLE) {
        double* data = static_cast<double*>(dataVar->data);

        // Some MNIST .mat files store pixels as double in [0,255].
        // Detect that case and normalize to [0,1] to avoid sigmoid saturation.
        double maxPixel = 0.0;
        for (size_t idx = 0; idx < rows * cols; ++idx) {
            if (data[idx] > maxPixel) {
                maxPixel = data[idx];
            }
        }
        const double scale = (maxPixel > 1.0) ? 255.0 : 1.0;

        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                images[c][r] = data[r + c * rows] / scale;
    } else if (dataVar->class_type == MAT_C_UINT8) {
        uint8_t* data = static_cast<uint8_t*>(dataVar->data);
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                images[c][r] =
                    static_cast<double>(data[r + c * rows]) / 255.0; // Normalize to [0, 1]
    } else {
        std::cerr << "Unsupported data type: " << dataVar->class_type << std::endl;
        Mat_VarFree(dataVar);
        Mat_Close(dataset);
        return;
    }

    Mat_VarFree(dataVar);

    /* -------- Read labels -------- */
    matvar_t* labelVar = Mat_VarRead(dataset, "label");
    if (!labelVar) {
        std::cerr << "Cannot find variable 'label'\n";
        Mat_Close(dataset);
        return;
    }

    if (!labelVar->data) {
        std::cerr << "Label variable has no data\n";
        Mat_VarFree(labelVar);
        Mat_Close(dataset);
        return;
    }

    double* labels_raw = static_cast<double*>(labelVar->data);
    labels = std::vector<std::vector<double>>(cols, std::vector<double>(10, 0));

    for (size_t i = 0; i < cols; ++i) {
        labels[i][static_cast<int>(labels_raw[i])] = 1;
    }

    Mat_VarFree(labelVar);
    Mat_Close(dataset);
}

// Given a canvas with a certain number drawn, finds the bounding box of the drawing, downscales it
// to 20x20 and centers it on a buffer that's the same size of the canvas
uint32_t* center_canvas_image(Canvas* canvas) {
    uint32_t* buffer = new uint32_t[canvas->getWidth() * canvas->getHeight()];
    std::memset(buffer, 0, canvas->getWidth() * canvas->getHeight() * sizeof(uint32_t));

    int minX = canvas->getWidth(), maxX = 0;
    int minY = canvas->getHeight(), maxY = 0;

    // Find bounding box
    for (int y = 0; y < canvas->getHeight(); ++y) {
        for (int x = 0; x < canvas->getWidth(); ++x) {
            if (canvas->getValue(x, y) != 0) {
                if (x < minX)
                    minX = x;
                if (x > maxX)
                    maxX = x;
                if (y < minY)
                    minY = y;
                if (y > maxY)
                    maxY = y;
            }
        }
    }

    // If no drawing found, return empty buffer
    if (maxX < minX || maxY < minY) {
        return buffer;
    }

    int num_w = maxX - minX + 1;
    int num_h = maxY - minY + 1;

    // Calculate scaling to fit in 20x20
    float scaling_factor = 1.0f;
    int scaled_w = num_w;
    int scaled_h = num_h;

    if (num_w > 20 || num_h > 20) {
        if (num_w > num_h) {
            scaling_factor = (float)num_w / 20.0f;
            scaled_w = 20;
            scaled_h = (int)((float)num_h / scaling_factor);
        } else {
            scaling_factor = (float)num_h / 20.0f;
            scaled_h = 20;
            scaled_w = (int)((float)num_w / scaling_factor);
        }
    }

    // Center position in output buffer
    int scaled_x = (canvas->getWidth() / 2) - (scaled_w / 2);
    int scaled_y = (canvas->getHeight() / 2) - (scaled_h / 2);

    // Downscale using area averaging (written with claude)
    for (int dst_y = 0; dst_y < scaled_h; ++dst_y) {
        for (int dst_x = 0; dst_x < scaled_w; ++dst_x) {
            // Map back to source coordinates
            float src_x_start = minX + (dst_x * scaling_factor);
            float src_y_start = minY + (dst_y * scaling_factor);
            float src_x_end = src_x_start + scaling_factor;
            float src_y_end = src_y_start + scaling_factor;

            // Average all pixels in source region
            float sum = 0.0f;
            int count = 0;

            for (int src_y = (int)src_y_start; src_y < (int)src_y_end && src_y <= maxY; ++src_y) {
                for (int src_x = (int)src_x_start; src_x < (int)src_x_end && src_x <= maxX;
                     ++src_x) {
                    uint32_t pixel = canvas->getValue(src_x, src_y);
                    // Extract alpha channel (or use full value if monochrome)
                    float value = (float)(pixel & 0xFF) / 255.0f;
                    sum += value;
                    count++;
                }
            }

            if (count > 0) {
                float avg = sum / count;
                uint8_t gray = (uint8_t)(avg * 255.0f);
                uint32_t color = (gray << 24) | (gray << 16) | (gray << 8) | gray; // ABGR format
                buffer[(scaled_y + dst_y) * canvas->getWidth() + (scaled_x + dst_x)] = color;
            }
        }
    }

    return buffer;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Not enough arguments, use --in=<.mat file> and --out<.bin file>" << std::endl;
        return 0;
    }
    std::string input_path;
    std::string output_path;
    for (size_t i = 0; i < argc; ++i) {
        std::string param(argv[i]);
        if (param.find("--in=") != std::string::npos) {
            input_path = param.substr(param.find("=") + 1, param.size());
        } else if (param.find("--out=") != std::string::npos) {
            output_path = param.substr(param.find("=") + 1, param.size());
        }
    }
    if (input_path.find(".mat") != std::string::npos && output_path.size() != 0) {
        std::vector<std::vector<double>> images;
        std::vector<std::vector<double>> labels;
        NeuralNetwork nenu = NeuralNetwork({784, 512, 10});

        load_data(input_path, images, labels);
        if (images.size() == 0 && labels.size() == 0) {
            std::cerr << "Error loading data" << std::endl;
            return 0;
        }
        std::cout << "Data loaded" << std::endl;

        NeuralNetwork::TrainResponse resp = nenu.train(images, labels, 0.8, 50, 50, 0.09, 1);
        std::cout << resp.averageCost << std::endl;
        std::cout << resp.maxCost << std::endl;
        std::cout << resp.minCost << std::endl;
        std::cout << resp.hitPercentage << std::endl;

        nenu.saveWeights(output_path);
    } else if (input_path.find(".bin") != std::string::npos) {
        NeuralNetwork* nenu = new NeuralNetwork();
        nenu->loadWeights(input_path);
        SDL_Window* window = SDL_CreateWindow("Test", 1024, 768, SDL_WINDOW_RESIZABLE);
        SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
        Canvas* canvas = new Canvas(28, 28, renderer);
        int wh, ww;
        SDL_GetWindowSize(window, &ww, &wh);
        SDL_FRect* rect = new SDL_FRect(0, 0, ww, wh);
        bool exit = false;
        bool mousePressed = false;
        while (!exit) {
            SDL_RenderClear(renderer);
            canvas->render(renderer, rect);
            SDL_RenderPresent(renderer);

            SDL_Event event;
            while (SDL_PollEvent(&event) == 1) {
                switch (event.type) {
                case SDL_EVENT_MOUSE_BUTTON_DOWN: {
                    if (event.button.button == SDL_BUTTON_LEFT) {

                        mousePressed = true;
                    }
                    break;
                }
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (event.button.button == SDL_BUTTON_LEFT) {

                        mousePressed = false;
                    }
                    break;
                case SDL_EVENT_MOUSE_MOTION: {
                    SDL_FPoint p = SDL_FPoint(event.motion.x, event.motion.y);
                    if (mousePressed && SDL_PointInRectFloat(&p, rect)) {
                        int canvasX = (int)(((event.motion.x - rect->x) / rect->w) * 28);
                        int canvasY = (int)(((event.motion.y - rect->y) / rect->h) * 28);
                        canvas->setPixel(canvasX, canvasY, 0xFFFFFFFF);
                        canvas->setPixel(canvasX + 1, canvasY + 1, 0xFFFFFFFF);
                        canvas->setPixel(canvasX, canvasY + 1, 0xFFFFFFFF);
                        canvas->setPixel(canvasX + 1, canvasY, 0xFFFFFFFF);
                    }
                    break;
                }
                case SDL_EVENT_KEY_DOWN: {
                    if (event.key.key == SDLK_C) {
                        canvas->clear();
                        break;
                    }
                    if (event.key.key == SDLK_RETURN) {
                        uint32_t* buf = center_canvas_image(canvas);
                        Matrix<double> in(1, 28 * 28);
                        for (size_t i = 0; i < (28 * 28); i++) {
                            double color = (double)(*(buf + i)) / 0xFFFFFFFF;
                            in.setValue(0, i, color);
                        }
                        Matrix<double> out = nenu->foward(in);
                        for (size_t i = 0; i < out.getHeight(); i++) {
                            std::cout << i << ": " << std::fixed << std::setprecision(6)
                                      << out.getValue(0, i) << std::endl;
                            ;
                        }
                        std::cout << std::endl;
                        std::cout << "----------------" << std::endl;
                        std::cout << std::endl;
                        delete buf;
                        break;
                    }
                    break;
                }
                case SDL_EVENT_QUIT: {
                    exit = true;
                    break;
                }
                default:
                    break;
                }
            }
        }
    }

    return 0;
}