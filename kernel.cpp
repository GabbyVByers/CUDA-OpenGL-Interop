
#include "opengl.h"

__global__ void pixelKernel(uchar4* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = -1;

    if ((x < width) && (y < height))
        int index = y * width + x;

    if (index == -1)
        return;

    pixels[index] = make_uchar4(255, 0, 255, 255);
}

void InteropOpenGL::executePixelKernel()
{
    uchar4* pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&pixels, &size, cudaPBO);
    pixelKernel <<<grid, block>>> (pixels, screenWidth, screenHeight);
    cudaDeviceSynchronize();
}

