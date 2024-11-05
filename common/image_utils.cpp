// image_utils.cpp

#include "image_utils.h"
#include "stb_image_write.h" // https://github.com/nothings/stb/blob/master/stb_image_write.h
#include <cmath>

RGBImage::RGBImage(int w, int h) : width(w), height(h), data(3 * w * h, 0) {}

void RGBImage::setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
    // No olvidar validar lo limites
    if (x < 0 || x >= width || y < 0 || y >= height) return;

    int index = 3 * (y * width + x); // Recordemos que trabajamos con un vector plano

    data[index] = r;
    data[index + 1] = g;
    data[index + 2] = b;
}

RGBImage convertToRGB(int width, int height, const std::vector<unsigned char> &sadImage) {
    RGBImage happyImage(width, height);

    // Convertir la imagen en escala de grises a RGB
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char gray = sadImage[y * width + x];
            happyImage.setPixel(x, y, gray, gray, gray);
        }
    }

    return happyImage; // Happy in appearance cause all the colors are the same XD
}

/// Draw the line in the RGB image using the EQUATION
/// r = x * cos(theta) + y * sin(theta)
/// This is called the infamous Hough Transform
void drawLine(RGBImage &image, float r, float theta, unsigned char r_col, unsigned char g_col, unsigned char b_col) {
    // Ok, we gonna assume theta is already in radians

    // Determine the extremes
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    // Calculate the edges
    float x0 = r * cos_theta;
    float y0 = r * sin_theta;

    float w = image.width;
    float h = image.height;

    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    if (sin_theta != 0) // This means the line is not vertical
    {
        x1 = 0;
        y1 = (r - x1 * cos_theta) / sin_theta;
        x2 = w;
        y2 = (r - x2 * cos_theta) / sin_theta;
    } else {
        x1 = r / cos_theta;
        y1 = 0;
        x2 = x1
        y2 = h;
    }

    // Actually de reverse of yCoord and xCoord are required
    int yCent = h / 2;
    int xCent = w / 2;

    int px1 = static_cast<int>(x1 + xCent); // Using static cast to avoid warnings and safe conversion
    int py1 = static_cast<int>(yCent - y1); // REMEMBER we previously inverted the y axis
    int px2 = static_cast<int>(x2 + xCent);
    int py2 = static_cast<int>(yCent - y2);

    // Si si, hay otros pero usaremos Bresenham para arrancar, luego vemos
    // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    // Bresenham's line algorithm
    int dx = std::abs(px2 - px1);
    int dy = std::abs(py2 - py1);
    int sx = (px1 < px2) ? 1 : -1; // Dios bendiga a quien puso el operador ternario en C++
    int sy = (py1 < py2) ? 1 : -1;
    int err = dx - dy;

    int x = px1;
    int y = py1;

    while (true) {
        image.setPixel(x, y, r_col, g_col, b_col);
        if (x == px2 && y == py2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx) { err += dx; y += sy; }
    }
}

// Save the image in some human readable format
bool saveImage(const RGBImage &image, const std::string &filename) {
    int channels = 3; // RGB
    int stride_in_bytes = channels * image.width;

    // Determine the format according to the file extension
    if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".png") {
        return stbi_write_png(filename.c_str(), image.width, image.height, channels, image.data.data(), stride) != 0;
    }
    else if (filename.size() >= 4 && (filename.substr(filename.size() - 4) == ".jpg" || filename.substr(filename.size() - 5) == ".jpeg")) {
        return stbi_write_jpg(filename.c_str(), image.width, image.height, channels, image.data.data(), 100) != 0; // Calidad 100 >.<
    }
    else {
        // Formato no soportado
        return false;
    }
}
