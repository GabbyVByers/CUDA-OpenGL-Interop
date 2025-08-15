
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

__global__ void kernel(int* device_a, const int* device_b, const int* device_c, const int size)
{
	int index = threadIdx.x;
	if (index >= size)
		return;
	device_a[index] = device_b[index] + device_c[index];
}

void runTestKernel()
{
	int size = 15;

	int* host_a = new int[size];
	int* host_b = new int[size];
	int* host_c = new int[size];

	int* device_a = nullptr;
	int* device_b = nullptr;
	int* device_c = nullptr;

	cudaMalloc((void**)&device_a, sizeof(int) * size);
	cudaMalloc((void**)&device_b, sizeof(int) * size);
	cudaMalloc((void**)&device_c, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		host_a[i] = 0;
		host_b[i] = i;
		host_c[i] = i * 10;
	}

	cudaMemcpy(device_a, host_a, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_c, host_c, sizeof(int) * size, cudaMemcpyHostToDevice);

	kernel <<<1, size>>> (device_a, device_b, device_c, size);
	cudaDeviceSynchronize();

	cudaMemcpy(host_a, device_a, sizeof(int) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, device_b, sizeof(int) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, device_c, sizeof(int) * size, cudaMemcpyDeviceToHost);

	std::cout << "A = B + C: "; for (int i = 0; i < size; i++) { std::cout << host_a[i] << " "; }
	std::cout <<       "\nB: "; for (int i = 0; i < size; i++) { std::cout << host_b[i] << " "; }
	std::cout <<       "\nC: "; for (int i = 0; i < size; i++) { std::cout << host_c[i] << " "; }

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
}

const char* vertexShaderSrc = R"glsl(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
    }
)glsl";

const char* fragmentShaderSrc = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 0.3, 1.0, 1.0);
    }
)glsl";

int main()
{
    runTestKernel();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Triangle", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertexShaderSrc, nullptr);
    glCompileShader(vertShader);
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragmentShaderSrc, nullptr);
    glCompileShader(fragShader);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    float triangleVerts[] =
    {
        0.0f,  0.5f, 0.0f,
       -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVerts), triangleVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window))
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

