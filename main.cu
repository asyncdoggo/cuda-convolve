#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void convolutionCUDA(float* inputImage, float* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float kernel[3][3] = { {-1, -1, -1},
                              {-1,  8, -1},
                              {-1, -1, -1} };

        float sum = 0.0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int imageX = min(max(x + i, 0), width - 1);
                int imageY = min(max(y + j, 0), height - 1);
                sum += kernel[i + 1][j + 1] * inputImage[imageY * width + imageX];
            }
        }
        outputImage[y * width + x] = sum;
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
    cudaMalloc(&d_inputImage, imageSize * sizeof(float));
    cudaMalloc(&d_outputImage, imageSize * sizeof(float));
    cudaMemcpy(d_inputImage, inputImage.data, imageSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto startCUDA = std::chrono::high_resolution_clock::now();
    convolutionCUDA << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, width, height);
    cudaDeviceSynchronize();
    auto endCUDA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
    std::cout << "CUDA convolution time: " << cudaTime.count() << " seconds" << std::endl;

    cudaMemcpy(outputImageCUDA.data, d_outputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    cv::imwrite("output_image_sequential.jpg", outputImageSequential);
    cv::imwrite("output_image_CUDA.jpg", outputImageCUDA);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
