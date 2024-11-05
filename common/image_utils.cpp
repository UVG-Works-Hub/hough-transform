#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // https://github.com/nothings/stb/blob/master/stb_image_write.h

#include "image_utils.h"
#include <cmath>
#include <algorithm>  // For std::clamp


// Implementación de RGBImage
RGBImage::RGBImage(int w, int h) : width(w), height(h), data(3 * w * h, 0) {}

void RGBImage::setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
    // No olvidar validar los límites
    if (x < 0 || x >= width || y < 0 || y >= height) return;

    int index = 3 * (y * width + x); // Recordemos que trabajamos con un vector plano

    data[index] = r;
    data[index + 1] = g;
    data[index + 2] = b;
}

// Convertir imagen en escala de grises a RGB
RGBImage convertToRGB(int width, int height, const std::vector<unsigned char> &grayImage) {
    RGBImage rgb(width, height);

    // Convertir la imagen en escala de grises a RGB
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char gray = grayImage[y * width + x];
            rgb.setPixel(x, y, gray, gray, gray);
        }
    }

    return rgb; // "Happy" in appearance because all the colors are the same XD
}

/// Draw the line in the RGB image using the EQUATION
/// r = x * cos(theta) + y * sin(theta)
/// This is called the infamous Hough Transform
void drawLine(RGBImage &image, float r, float theta, unsigned char r_col, unsigned char g_col, unsigned char b_col) {
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float w = image.width;
    float h = image.height;

    int yCent = h / 2;
    int xCent = w / 2;

    // Adjust r to image coordinates by incorporating the center offsets
    float r_img = r + xCent * cos_theta + yCent * sin_theta;

    float x1, y1, x2, y2;

    if (sin_theta != 0) {
        // Intersection with the left border (x = 0)
        x1 = 0;
        y1 = (r_img - x1 * cos_theta) / sin_theta;

        // Intersection with the right border (x = w - 1)
        x2 = w - 1;
        y2 = (r_img - x2 * cos_theta) / sin_theta;
    } else {
        // Vertical line: Intersection with the top border (y = 0)
        x1 = r_img / cos_theta;
        y1 = 0;

        // Intersection with the bottom border (y = h - 1)
        x2 = x1;
        y2 = h - 1;
    }

    // Convert to integer pixel positions
    int px1 = static_cast<int>(x1);
    int py1 = static_cast<int>(y1);
    int px2 = static_cast<int>(x2);
    int py2 = static_cast<int>(y2);

    // Ensure px1, py1, px2, py2 are within image bounds
    px1 = std::clamp(px1, 0, image.width - 1);
    py1 = std::clamp(py1, 0, image.height - 1);
    px2 = std::clamp(px2, 0, image.width - 1);
    py2 = std::clamp(py2, 0, image.height - 1);

    // Bresenham's line algorithm to draw the line
    int dx = abs(px2 - px1);
    int dy = abs(py2 - py1);
    int sx = (px1 < px2) ? 1 : -1;
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



// Guardar la imagen en formato PNG o JPG usando stb_image_write
bool saveImage(const RGBImage &image, const std::string &filename) {
    int channels = 3; // RGB
    int stride_in_bytes = channels * image.width;

    // Determinar el formato según la extensión del archivo
    if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".png") {
        return stbi_write_png(filename.c_str(), image.width, image.height, channels, image.data.data(), stride_in_bytes) != 0;
    }
    else if (filename.size() >= 4 && (filename.substr(filename.size() - 4) == ".jpg" || filename.substr(filename.size() - 5) == ".jpeg")) {
        return stbi_write_jpg(filename.c_str(), image.width, image.height, channels, image.data.data(), 100) != 0; // Calidad 100
    }
    else {
        // Formato no soportado
        return false;
    }
}
