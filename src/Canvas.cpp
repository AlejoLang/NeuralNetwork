#include "../include/Canvas.hpp"
#include <cstring>
#include <iostream>

Canvas::Canvas(int w, int h, SDL_Renderer* renderer) {
    this->texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    SDL_SetTextureScaleMode(this->texture, SDL_SCALEMODE_PIXELART);
    this->buffer = new uint32_t[w * h];
    this->width = w;
    this->height = h;
}

uint32_t* Canvas::getBuffer() {
    return this->buffer;
}

void Canvas::setPixel(int x, int y, uint32_t color) {
    if (x > this->width || y > this->height) {
        std::cerr << "Invalida coords" << std::endl;
        return;
    }
    this->buffer[(y * this->width) + x] = color;
}

void Canvas::clear() {
    std::memset(this->buffer, 0, this->width * this->height * sizeof(uint32_t));
}

void Canvas::render(SDL_Renderer* renderer, SDL_FRect* rect) {
    SDL_UpdateTexture(this->texture, NULL, this->buffer, sizeof(uint32_t) * this->width);
    SDL_RenderTexture(renderer, this->texture, NULL, rect);
}

Canvas::~Canvas() {
    delete[] this->buffer;
    SDL_DestroyTexture(this->texture);
}