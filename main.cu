#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void convolutionCUDA(float* inputImage, float* outputImage, int width, int height) {
    // CUDA kernel to perform convolution

    int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of the pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of the pixel

    if (x < width && y < height) { // Check if the pixel is within the image boundaries

        // This is a kernel for edge detection
        float kernel[3][3] = { {-1, -1, -1},
                              {-1,  8, -1},
                              {-1, -1, -1} };

        
        float sum = 0.0; // Initialize the sum to zero, this is the value of the pixel after convolution
        for (int i = -1; i <= 1; i++) { // Loop over the kernel
            for (int j = -1; j <= 1; j++) {  
                int imageX = min(max(x + i, 0), width - 1); // Get the x coordinate of the pixel in the image
                int imageY = min(max(y + j, 0), height - 1); // Get the y coordinate of the pixel in the image
                sum += kernel[i + 1][j + 1] * inputImage[imageY * width + imageX]; // Convolution operation
            }
        }
        outputImage[y * width + x] = sum; // Store the value of the pixel in the output image
    }
}

void convolutionSequential(cv::Mat& inputImage, cv::Mat& outputImage) {
    float kernel[3][3] = { {-1, -1, -1},
                          {-1,  8, -1},
                          {-1, -1, -1} };
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            float sum = 0.0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int imageX = std::min(std::max(x + i, 0), inputImage.cols - 1);
                    int imageY = std::min(std::max(y + j, 0), inputImage.rows - 1);
                    sum += kernel[i + 1][j + 1] * inputImage.at<float>(imageY, imageX);
                }
            }
            outputImage.at<float>(y, x) = sum;
        }
    }
}

int main() {
    std::cout << "Enter the input image name:";
    std::string inputImageName;
    std::cin >> inputImageName;

    cv::Mat inputImage = cv::imread(inputImageName, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Unable to read input image." << std::endl;
        return -1;
    }

    cv::Mat outputImageSequential(inputImage.size(), CV_32FC1);
    cv::Mat outputImageCUDA(inputImage.size(), CV_32FC1);

    // Convert input image to floating point
    inputImage.convertTo(inputImage, CV_32FC1);

    // Sequential convolution
    auto startSeq = std::chrono::high_resolution_clock::now();
    convolutionSequential(inputImage, outputImageSequential);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqTime = endSeq - startSeq;
    std::cout << "Sequential convolution time: " << seqTime.count() << " seconds" << std::endl;

    // CUDA convolution
    int width = inputImage.cols; 
    int height = inputImage.rows;
    int imageSize = width * height;
    float* d_inputImage, * d_outputImage;
    cudaMalloc(&d_inputImage, imageSize * sizeof(float)); // Allocate memory on the GPU
    cudaMalloc(&d_outputImage, imageSize * sizeof(float)); // Allocate memory on the GPU
    cudaMemcpy(d_inputImage, inputImage.data, imageSize * sizeof(float), cudaMemcpyHostToDevice); // Copy input image to GPU

    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y); // Number of blocks

    auto startCUDA = std::chrono::high_resolution_clock::now();
    convolutionCUDA << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, width, height); // Call the CUDA kernel
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    auto endCUDA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
    std::cout << "CUDA convolution time: " << cudaTime.count() << " seconds" << std::endl;

    cudaMemcpy(outputImageCUDA.data, d_outputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost); // Copy output image to CPU

    cv::imwrite("output_image_sequential.jpg", outputImageSequential);
    cv::imwrite("output_image_CUDA.jpg", outputImageCUDA);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
