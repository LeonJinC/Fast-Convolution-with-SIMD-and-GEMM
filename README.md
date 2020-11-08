# Fast-Convolution-with-SIMD-and-GEMM
Fast Convolution with SIMD and GEMM; 
Application on Gaussian Blur;
结合SIMD和GEMM的快速卷积及其在高斯模糊中的应用

项目属性设置：

C/C++ -> 优化 -> 优化：使速度最大化/O2

C/C++ -> 代码生成 -> 启用增强指令集：高级矢量扩展2 /arch:AVX2

C/C++ -> 语言 -> OpenMP支持：是 /openmp

需要的话另外配置opencv的环境

需要用的一些较为重要的库：

omp.h ：openMP多线程库，是对多线程技术的抽象封装

immintrin.h ： SIMD单指令多数据C++库，是对SIMD的抽象封装，其中的api函数和数据类型经过C++编译之后可以生成支持AVX2的汇编语句


# 实验结果0：GEMM方法比较
![实验结果0](https://github.com/LeonJinC/Fast-Convolution-with-SIMD-and-GEMM/blob/main/RESULTS0GEMM.jpg)

# 实验结果1：输入图像128x128高斯核3x3
![实验结果1](https://github.com/LeonJinC/Fast-Convolution-with-SIMD-and-GEMM/blob/main/RESULTS1Fast_Convolution3.jpg)

# 实验结果2：输入图像128x128高斯核7x7
![实验结果2](https://github.com/LeonJinC/Fast-Convolution-with-SIMD-and-GEMM/blob/main/RESULTS2Fast_Convolution7.jpg)

# 实验结果3：输入图像512x512高斯核3x3
![实验结果3](https://github.com/LeonJinC/Fast-Convolution-with-SIMD-and-GEMM/blob/main/RESULTS3Image.jpg)

# 实验结果4：输入图像8x8高斯核3x3
![实验结果4](https://github.com/LeonJinC/Fast-Convolution-with-SIMD-and-GEMM/blob/main/RESULTS4Matrix.jpg)

hello


