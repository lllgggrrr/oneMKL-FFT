# oneMKL FFT

## 简介

本代码旨在解释一个使用C代码实现的性能比较示例，比较了两个不同库，即FFTW和英特尔数学核心库（OneMKL），在执行快速傅里叶变换（FFT）方面的性能差异。代码生成随机数据并利用这两个库来计算FFT，进而比较它们的执行时间和计算结果。



## OneMKL介绍

Intel Math Kernel Library（OneMKL）是由英特尔开发的高性能数学库，专门用于优化数值计算任务。它提供了一系列数学和线性代数函数，旨在加速科学计算、工程仿真、数据分析等应用领域的计算任务。OneMKL 可以帮助开发人员快速完成高性能的计算任务。



## 代码结构

以下是代码的主要结构和功能的解释：

1. **核心头文件：** FFTW头文件 `<fftw3.h>`、英特尔数学核心库头文件 `<mkl.h>`

2. **生成随机数据：** `make_random_data()` 函数利用英特尔数学核心库（MKL）生成均匀分布的随机数。它使用了 OneMKL 提供的随机数生成函数生成(0.0f,1.0f)的随机数。

   ```c
   /*
   * 生成Row行Col列的随机数
   */
   void make_random_data() {
       VSLStreamStatePtr stream;
       vslNewStream(&stream, VSL_BRNG_MT19937, 1);
       vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N1 * N2, random_data, 0.0f, 1.0f);
       vslDeleteStream(&stream);
   }
   ```

   

3. **使用FFTW进行FFT计算：** `cal_complex_fftwf()` 函数执行 FFTW 库的快速傅里叶变换计算。

   ```c
   /*
   * 普通的快速傅里叶变换
   * 此处参照文档
   * 返回使用时间
   */
   double cal_complex_fftwf() {
       fftwf_output_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N1 * (N2 / 2 + 1));
       fftwf_plan fftwf_plan = fftwf_plan_dft_r2c_2d(N1, N2, random_data, fftwf_output_data, FFTW_ESTIMATE);
       
       LARGE_INTEGER start_time, end_time;
       QueryPerformanceCounter(&start_time);
       // 执行计算
       fftwf_execute(fftwf_plan);
       QueryPerformanceCounter(&end_time);
       // 销毁计划
       fftwf_destroy_plan(fftwf_plan);
       return (double)(end_time.QuadPart - start_time.QuadPart);
   }
   ```

   

4. **使用OneMKL进行FFT计算：** `cal_complex_fft_MKL()` 函数执行英特尔数学核心库（MKL）的快速傅里叶变换计算。

   ```c
   /*
   * OneMKL 傅里叶变换
   */
   double cal_complex_fft_MKL() {
       MKL_LONG dim_sizes[2] = { N1, N2 };
       MKL_LONG rs[3] = { 0, N1, 1 };
       MKL_LONG cs[3] = { 0, N2 / 2 + 1, 1 };
       DFTI_DESCRIPTOR_HANDLE handle;
       // 设置一些计算参数
       DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 2, dim_sizes);
       DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
       DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
       DftiSetValue(handle, DFTI_INPUT_STRIDES, rs);
       DftiSetValue(handle, DFTI_OUTPUT_STRIDES, cs);
       DftiCommitDescriptor(handle);
       mkl_output_data = (MKL_Complex8*)malloc(sizeof(MKL_Complex8) * N1 * (N2 / 2 + 1));
       LARGE_INTEGER start_time, end_time;
       QueryPerformanceCounter(&start_time);
       // 执行计算
       DftiComputeForward(handle, random_data, mkl_output_data);
       QueryPerformanceCounter(&end_time);
       DftiFreeDescriptor(&handle);
       return (double)(end_time.QuadPart - start_time.QuadPart);
   }
   ```

   

5. **主函数：**执行1000次循环计算（生成随机数、FFTW、OneMKL），对比FFTW和oneMKL的计算结果，输出平均运行时间。

   ```c
   int main() {
       double fftw_total_time = 0.0;
       double mkl_total_time = 0.0;
   
       // 测试一千次取平均
       for (int times = 0; times < TOTAL_TIMES; times++) {
   
           make_random_data();
           fftw_total_time += cal_complex_fftwf();
           mkl_total_time += cal_complex_fft_MKL();
   
           int flag = 1;
           for (int i = 0; i < (N2 / 2 + 1) * N1; i++) {
               if (fabs(mkl_output_data[i].real - fftwf_output_data[i][0]) > 1e-6) {
                   flag = 0;
                   break;
               }
               if (fabs(mkl_output_data[i].imag - fftwf_output_data[i][1]) > 1e-6) {
                   flag = 0;
                   break;
               }
           }
   
   
           if (flag) {
               printf("%d RESULT CORRECT!\n", times);
           }
           else {
               printf("%d RESULT ERROR!\n", times);
           }
   
           fftwf_free(fftwf_output_data);
           free(mkl_output_data);
       }
   
       printf("FFTWF AVERAGE RUN TIME: %f US \n", fftw_total_time / TOTAL_TIMES);
       printf("OneMKL FFT AVERAGE RUN TIME: %f US \n", mkl_total_time / TOTAL_TIMES);
   
       return 0;
   }
   ```

   

## 使用方法

1. 安装所需的库：确保已经安装了 FFTW 和英特尔数学核心库（OneMKL）。

2. 使用 Visual Studio 2022 打开项目。

3. 配置项目的属性，修改OneMKL设置选项。

4. 使用 Visual Studio 进行编译和调试。

   

## 结论

本文档提供的代码示例演示了使用不同库进行FFT计算的性能比较。通过生成随机数据并进行FFT计算，我们可以了解不同库的性能表现和计算结果的准确性。从中我们可以发现，Intel OneMKL相较于FFTW在保证计算正确率的情况下，运算速度有较大的提升。