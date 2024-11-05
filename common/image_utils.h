// image_utils.h

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <string>

// Estructura para representar una imagen RGB
struct RGBImage {
    int width;
    int height;
    std::vector<unsigned char> data; // Formato RGB (3 bytes por píxel)

    RGBImage(int w, int h);
    void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
};

// Funciones utilitarias
RGBImage convertToRGB(int width, int height, const std::vector<unsigned char> &grayImage);
void drawLine(RGBImage &image, float r, float theta, unsigned char r_color, unsigned char g_color, unsigned char b_color);
bool saveImage(const RGBImage &image, const std::string &filename);

#endif // IMAGE_UTILS_H
