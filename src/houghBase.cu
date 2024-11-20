/*
 ============================================================================
 Autor         : G. Barlas
 Version       : 1.0
 Ultima modificacion : Diciembre 2014
 Licencia      : Liberado bajo la GNU GPL 3.0
 Descripcion   :
 Compilar usando : make
 ============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "../common/pgm.h"
#include "../common/image_utils.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// La funcion CPU devuelve un puntero al acumulador
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];                // el acumulador, conteo de pixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); // inicializar en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)     // por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) // si pasa umbral, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;                       // y-coord tiene que estar invertido
                float theta = 0;                              // angulo actual
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) // agregar 1 a todas las lineas en ese pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    if (rIdx >= 0 && rIdx < rBins)
                    {                                       // Validar indices
                        (*acc)[rIdx * degreeBins + tIdx]++; // +1 para este radio r y este theta
                    }
                    theta += radInc;
                }
            }
        }
}

//*****************************************************************
// Kernel GPU. Un hilo por pixel de la imagen es lanzado.
// La memoria del acumulador necesita ser asignada por el host en memoria global
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return; // Corregido: >= en lugar de >

    int xCent = w / 2;
    int yCent = h / 2;

    // Explicacion:
    // El calculo de las coordenadas `xCoord` y `yCoord` se realiza para modificar el sistema de coordenadas de la imagen.
    // En lugar de basarse en el sistema de ubicacion de pixeles tradicional, donde el origen (0,0) esta en la esquina superior izquierda,
    // estas coordenadas se calculan respecto al centro de la imagen. `xCoord` se obtiene restando la mitad del ancho de la imagen a la coordenada X del pixel,
    // y `yCoord` se obtiene restando la coordenada Y del pixel de la mitad de la altura de la imagen, invirtiendo asi el eje Y, pues normalmente el eje Y crece hacia abajo.
    // Este cambio es util para los pasos posteriores, ya que facilita el calculo de la distancia de cada punto a una recta en el espacio de parametros (r, theta).

    int xCoord = gloID % w - xCent;
    int yCoord = gloID / w - yCent;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

//*****************************************************************
int main(int argc, char **argv)
{
    // Variables para argumentos
    std::string inputFilename;
    std::string outputFilename = "output.png"; // Valor por defecto
    float threshold = -1.0f;                   // Valor por defecto que indica calculo automatico

    // Procesar argumentos
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output")
        {
            if (i + 1 < argc)
            {
                outputFilename = argv[++i];
            }
            else
            {
                printf("Error: Se requiere un valor despues de %s\n", arg.c_str());
                return 1;
            }
        }
        else if (arg == "-t" || arg == "--threshold")
        {
            if (i + 1 < argc)
            {
                threshold = atof(argv[++i]);
            }
            else
            {
                printf("Error: Se requiere un valor despues de %s\n", arg.c_str());
                return 1;
            }
        }
        else
        {
            inputFilename = arg;
        }
    }

    if (inputFilename.empty())
    {
        printf("Uso: %s <imagen.pgm> [--output <salida.png/jpg>] [--threshold <valor>]\n", argv[0]);
        return 1;
    }

    int i;

    PGMImage inImg(inputFilename);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float *d_Cos;
    float *d_Sin;

    cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // Calculo en CPU
    CPU_HoughTran(inImg.pixels.data(), w, h, &cpuht);

    // Precalcular tablas de cosenos y senos
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // Configurar y copiar datos del host al device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels.data(); // h_in contiene los pixeles de la imagen

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Medicion de tiempo con eventos CUDA
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Registrar el evento al inicio del kernel (justo antes de la llamada al kernel)
    cudaEventRecord(startEvent, 0); // 0 es el stream por defecto

    // Configuracion de ejecucion usa una grilla 1-D de bloques 1-D, cada uno con 256 hilos
    // 1 hilo por pixel
    int blockNum = (w * h + 255) / 256; // Corregido: asegura cubrir todos los hilos
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // Registrar el evento de parada
    cudaEventRecord(stopEvent, 0);

    // Sincronizar el evento de parada
    cudaEventSynchronize(stopEvent); // Esperar a que el evento sea registrado!

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent); // Esto retorna en el primer argumento el tiempo entre los dos eventos

    // Destruir los eventos, como buena practica
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Obtener resultados del device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Comparar resultados de CPU y GPU
    bool mismatch = false;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
        {
            printf("Diferencia en calculo en indice %i: CPU=%i, GPU=%i\n", i, cpuht[i], h_hough[i]);
            mismatch = true;
            break; // Puedes detenerte en el primer error encontrado
        }
    }

    if (!mismatch)
    {
        printf("Los resultados de la Transformada de Hough en CPU y GPU coinciden.\n");
    }
    else
    {
        printf("Los resultados de la Transformada de Hough en CPU y GPU no coinciden.\n");
    }

    printf("Tiempo de Kernel (ms):: %f\n", milliseconds);

    // Procesar los resultados (AKA acumulador para detectar lineas)

    // Encontrar el valor maximo en el acumulador para definir el umbral si es necesario
    int max_acc = 0;
    long total = 0;

    for (int idx = 0; idx < degreeBins * rBins; idx++)
    {
        if (h_hough[idx] > max_acc)
        {
            max_acc = h_hough[idx];
        }
        total += h_hough[idx];
    }

    // Definir el umbral
    if (threshold < 0.0f)
    {
        // Por ejemplo, umbral = promedio + 2 * desviacion estandar (Como lo dijo el profesor)
        float average = (float)total / (degreeBins * rBins);
        float variance = 0.0f;

        for (int idx = 0; idx < degreeBins * rBins; idx++)
        {
            variance += (h_hough[idx] - average) * (h_hough[idx] - average); // Varianza
        }
        float std_dev = sqrt(variance / (degreeBins * rBins)); // Desviacion estandar
        threshold = average + 2 * std_dev;

        printf("Umbral Computado: %f (Promedio: %f, Desviacion Estandar: %f)\n", threshold, average, std_dev);
    }

    printf("Umbral para dibujar lineas: %f\n", threshold);

    struct Line
    {
        float r;
        float theta;
        int weight;
    };

    std::vector<Line> lines;

    // Encontrar lineas
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int weight = h_hough[rIdx * degreeBins + tIdx];
            if (weight >= threshold)
            {
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;          // Corregido: consistente con CPU_HoughTran
                lines.push_back(Line{r, theta, weight}); // Recuerda, r es la distancia desde el centro de la imagen y theta es el angulo
            }
        }
    }

    printf("Se encontraron %lu lineas (usando umbral %f)\n", lines.size(), threshold);

    // Finalmente, usar nuestra pequeña libreria para dibujar las lineas
    RGBImage rgbImage = convertToRGB(w, h, inImg.pixels);

    // Dibujar las lineas detectadas en color arbitrario (IDEA es luego hacer esto un parametro)
    for (const Line &line : lines)
    {
        drawLine(rgbImage, line.r, line.theta, 66, 245, 233);
    }

    // Guardar la imagen
    if (saveImage(rgbImage, outputFilename))
    {
        printf("Imagen guardada en %s\n", outputFilename.c_str());
    }
    else
    {
        printf("Error al guardar la imagen en %s\n", outputFilename.c_str());
    }

    printf("Hecho!\n");

    // limpieza
    // Liberar memoria del host
    free(cpuht);
    free(h_hough);
    free(pcCos);
    free(pcSin);

    // Liberar memoria del device
    cudaFree(d_in);
    cudaFree(d_hough);
    cudaFree(d_Cos);
    cudaFree(d_Sin);

    return 0;
}
