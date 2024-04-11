# CUDA Convolution


# Install opencv from
[https://github.com/opencv/opencv/releases](https://github.com/opencv/opencv/releases)

# Install CUDA from
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

# Compile the code
```bash
nvcc main.cu -o main -I "D:\tools\opencv\build\include" -L "D:\tools\opencv\build\x64\vc15\lib" -lopencv_world451 -lopencv_world451d
```

# Copy the opencv_worldxxx.dll and opencv_worldxxxd.dll from
```bash
<opencv_installation>\opencv\build\x64\vc15\bin
```

# Run the code
```bash
main.exe
```
