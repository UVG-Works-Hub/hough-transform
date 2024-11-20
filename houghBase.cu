/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include "common/image_utils.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];                // el acumulador, conteo de pixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); // init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)     // por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) // si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;                       // y-coord has to be reversed
                float theta = 0;                              // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) // add 1 to all lines in that pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    if (rIdx >= 0 && rIdx < rBins)
                    {                                       // Validar índices
                        (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                    }
                    theta += radInc;
                }
            }
        }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
// TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accumulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return; // Corregido: >= en lugar de >

    int xCent = w / 2;
    int yCent = h / 2;

    // Explicación:
    // El cálculo de las coordenadas `xCoord` y `yCoord` se realiza para modificar el sistema de coordenadas de la imagen.
    // En lugar de basarse en el sistema de ubicación de píxeles tradicional, donde el origen (0,0) está en la esquina superior izquierda,
    // estas coordenadas se calculan respecto al centro de la imagen. `xCoord` se obtiene restando la mitad del ancho de la imagen a la coordenada X del píxel,
    // y `yCoord` se obtiene restando la coordenada Y del píxel de la mitad de la altura de la imagen, invirtiendo así el eje Y, pues normalmente el eje Y crece hacia abajo.
    // Este cambio es util para los pasos posteriores, ya que facilita el cálculo de la distancia de cada punto a una recta en el espacio de parámetros (r, θ).

    int xCoord = gloID % w - xCent;
    int yCoord = gloID / w - yCent;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            // TODO utilizar memoria constante para senos y cosenos
            // float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

    // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    // utilizar operaciones atomicas para seguridad
    // faltara sincronizar los hilos del bloque en algunos lados
}

//*****************************************************************
int main(int argc, char **argv)
{
    // Variables para argumentos
    std::string inputFilename;
    std::string outputFilename = "output.png"; // Valor por defecto
    float threshold = -1.0f;                   // Valor por defecto que indica cálculo automático

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
                printf("Error: Se requiere un valor después de %s\n", arg.c_str());
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
                printf("Error: Se requiere un valor después de %s\n", arg.c_str());
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

    // CPU calculation
    CPU_HoughTran(inImg.pixels.data(), w, h, &cpuht);

    // Precompute cos and sin tables
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

    // TODO eventualmente volver memoria global
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // Setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels.data(); // h_in contiene los pixeles de la imagen

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // CUDA events time measurement
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Register the event at the start of the kernel (just before the kernel call)
    cudaEventRecord(startEvent, 0); // 0 is the default stream

    // Execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    // 1 thread por pixel
    int blockNum = (w * h + 255) / 256; // Corregido: asegura cubrir todos los hilos
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // Register the stop event
    cudaEventRecord(stopEvent, 0);

    // Synchronize the stop event
    cudaEventSynchronize(stopEvent); // Wait for the event to be recorded!

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent); // This returns in the first argument the time between the two events
    // As the guide says, this has a resolution of approx 0.5 microseconds

    // Destroy the events, just as a good practice
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    bool mismatch = false;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
        {
            printf("Calculation mismatch at index %i: CPU=%i, GPU=%i\n", i, cpuht[i], h_hough[i]);
            mismatch = true;
            break; // Puedes detenerte en el primer error encontrado
        }
    }

    if (!mismatch)
    {
        printf("CPU and GPU Hough Transform results match.\n");
    }
    else
    {
        printf("CPU and GPU Hough Transform results do not match.\n");
    }

    printf("Kernel time execution (ms): %f\n", milliseconds);

    // Process the results (AKA accumulator to detect lines)

    // Find the max value in the accumulator to define the threshold if necessary
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

    // Define the threshold
    if (threshold < 0.0f)
    {
        // Por ejemplo, umbral = promedio + 2 * desviación estándar (Como lo dijo el profesor)
        float average = (float)total / (degreeBins * rBins);
        float variance = 0.0f;

        for (int idx = 0; idx < degreeBins * rBins; idx++)
        {
            variance += (h_hough[idx] - average) * (h_hough[idx] - average); // Varianza
        }
        float std_dev = sqrt(variance / (degreeBins * rBins)); // Desviación estándar
        threshold = average + 2 * std_dev;

        printf("Computed Threshold: %f (Average: %f, Std Dev: %f)\n", threshold, average, std_dev);
    }

    printf("Threshold for drawing lines: %f\n", threshold);

    struct Line
    {
        float r;
        float theta;
        int weight;
    };

    std::vector<Line> lines;

    // Find lines
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int weight = h_hough[rIdx * degreeBins + tIdx];
            if (weight >= threshold)
            {
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;          // Corregido: consistente con CPU_HoughTran
                lines.push_back(Line{r, theta, weight}); // Recuerda, r es la distancia desde el centro de la imagen y theta es el ángulo
            }
        }
    }

    printf("Found %lu lines (using threshold %f)\n", lines.size(), threshold);

    // Finally using our small library to draw the lines
    RGBImage rgbImage = convertToRGB(w, h, inImg.pixels);

    // Draw the detected lines in arbitrary color (IDEA is to then make this a parameter)
    for (const Line &line : lines)
    {
        drawLine(rgbImage, line.r, line.theta, 255, 0, 0); // Rojo
    }

    // Save the image
    if (saveImage(rgbImage, outputFilename))
    {
        printf("Image saved to %s\n", outputFilename.c_str());
    }
    else
    {
        printf("Error saving image to %s\n", outputFilename.c_str());
    }

    printf("Done!\n");

    // cleanup
    // Free host memory
    free(cpuht);
    free(h_hough);
    free(pcCos);
    free(pcSin);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_hough);
    cudaFree(d_Cos);
    cudaFree(d_Sin);

    return 0;
}
