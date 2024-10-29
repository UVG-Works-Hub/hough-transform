// common/pgm.h

#ifndef PGM_H
#define PGM_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

class PGMImage {
public:
    int x_dim;
    int y_dim;
    std::vector<unsigned char> pixels;

    PGMImage(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        // Leer el encabezado PGM
        std::getline(file, line);
        if (line != "P5") {
            throw std::runtime_error("Formato PGM no soportado (debe ser P5)");
        }

        // Leer posibles comentarios
        do {
            std::getline(file, line);
        } while (line[0] == '#');

        // Leer dimensiones
        std::stringstream ss(line);
        ss >> x_dim >> y_dim;

        // Leer el máximo valor de píxel (ignoramos, asumimos 255)
        std::getline(file, line);

        // Leer los datos de píxeles
        pixels.resize(x_dim * y_dim);
        file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

        if (!file) {
            throw std::runtime_error("Error leyendo los datos de píxeles");
        }

        file.close();
    }
};

#endif // PGM_H
