#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <mkl.h>
#include <math.h>
#include <windows.h>

#define N1 2048
#define N2 2048
#define TOTAL_TIMES 1000
//存储每一次的随机数
float random_data[N1 * N2];
fftwf_complex* fftwf_output_data;
MKL_Complex8* mkl_output_data;

/*
* 生成Row行Col列的随机数
*/
void make_random_data() {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N1 * N2, random_data, 0.0f, 1.0f);
    vslDeleteStream(&stream);
}


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
    fftwf_execute(fftwf_plan);
    QueryPerformanceCounter(&end_time);
    fftwf_destroy_plan(fftwf_plan);
    return (double)(end_time.QuadPart - start_time.QuadPart);
}

/*
* OneMKL 傅里叶变换
*/
double cal_complex_fft_MKL() {
    MKL_LONG dim_sizes[2] = { N1, N2 };
    MKL_LONG rs[3] = { 0, N1, 1 };
    MKL_LONG cs[3] = { 0, N2 / 2 + 1, 1 };
    DFTI_DESCRIPTOR_HANDLE handle;
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 2, dim_sizes);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(handle, DFTI_INPUT_STRIDES, rs);
    DftiSetValue(handle, DFTI_OUTPUT_STRIDES, cs);
    DftiCommitDescriptor(handle);
    mkl_output_data = (MKL_Complex8*)malloc(sizeof(MKL_Complex8) * N1 * (N2 / 2 + 1));
    LARGE_INTEGER start_time, end_time;
    QueryPerformanceCounter(&start_time);
    DftiComputeForward(handle, random_data, mkl_output_data);
    QueryPerformanceCounter(&end_time);
    DftiFreeDescriptor(&handle);
    return (double)(end_time.QuadPart - start_time.QuadPart);
}


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
