#pragma once
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_video.h>

class Canvas {
  private:
    SDL_Texture* texture;
    uint32_t* buffer;
    int width;
    int height;

  public:
    Canvas(int w, int h, SDL_Renderer* renderer);
    uint32_t* getBuffer();
    int getWidth();
    int getHeight();
    uint32_t getValue(int x, int y);
    void setPixel(int x, int y, uint32_t color);
    void clear();
    void render(SDL_Renderer* renderer, SDL_FRect* rect = NULL);
    ~Canvas();
};