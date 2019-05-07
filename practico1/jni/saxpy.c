#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <arm_neon.h>

void saxpy(int n, float a, float x[], float y[])
{
    double start_time, run_time;
    start_time = omp_get_wtime();
    for (int i = 0; i < n; ++i)
    {   
        y[i] = a * x[i] + y[i];
    }
    run_time = omp_get_wtime() - start_time;
    printf("serie: %f seconds con tamaño: %d\n", run_time, n);

}

void saxpyPar(int n, float32_t a[], float32_t x[], float32_t y[])
{
    double start_time, run_time;
    omp_set_num_threads(2);
    start_time = omp_get_wtime();
#pragma omp parallel
    {
        for (int i = 0; i < n; i=i+2)
        {
            //*y = vmla_f32(*y, *x, *a);
            float32x2_t y2 = vld1_f32(y);       // VLD1.32 {d0}, [r0]
            float32x2_t x2 = vld1_f32(x);       // VLD1.32 {d0}, [r0]
            float32x2_t a2 = vld1_f32(a);       // VLD1.32 {d0}, [r0]
            //y[i] = a[i] * x[i] + y[i];
            vst1_f32(y, vmla_f32(y2, x2, a2));  // VST1.32 {d0}, [r0]
            y+=2;
            x+=2;
            a+=2;
        }
    }
    run_time = omp_get_wtime() - start_time;
    printf("paralelo: %f seconds con tamaño: %d\n", run_time, n);
}

void llenar(float arr[],int tam){
    for(int i = 0; i < tam; i++){
        arr[i]=i;
    }
}

void llenarNeon(float32_t *array, int size){    
    int i;
    for (i = 0; i < size; i++){
         array[i] = i;
    }
}

void llenarNeonPendiente(float m, float32_t *arr, int tam){
    for(int i = 0; i < tam; i++){
        arr[i]=m;
    }
}

int main(int argc, char **argv)
{

    float m = 2.0;
    int size = 4;
    float *x =(float*) calloc(size, sizeof(float));
    float *y =(float*) calloc(size, sizeof(float));
    llenar(x,size);
    llenar(y,size);
    saxpy(size,m,x,y);
    size = size*m;
    x = (float*) calloc(size, sizeof(float));
    y = (float*) calloc(size, sizeof(float));
    saxpy(size,m,x,y);
    size = size*m;
    x = (float*) calloc(size, sizeof(float));
    y = (float*) calloc(size, sizeof(float));
    saxpy(size,m,x,y);

    m = 2.0;
    size = 4;
    float32_t aNeon[size], yNeon[size], xNeon[size];
    llenarNeonPendiente(m,aNeon,size);
    llenarNeon(yNeon,size);
    llenarNeon(xNeon,size);
    saxpyPar(size,aNeon,xNeon,yNeon);
    
    size = size*m;
    float32_t aNeon1[size], yNeon1[size], xNeon1[size];
    llenarNeonPendiente(m,aNeon1,size);
    llenarNeon(yNeon1,size);
    llenarNeon(xNeon1,size);
    saxpyPar(size,aNeon1,xNeon1,yNeon1);

    size = size*m;
    float32_t aNeon2[size], yNeon2[size], xNeon2[size];
    llenarNeonPendiente(m,aNeon2,size);
    llenarNeon(yNeon2,size);
    llenarNeon(xNeon2,size);
    saxpyPar(size,aNeon2,xNeon2,yNeon2);

}