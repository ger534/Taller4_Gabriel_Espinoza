#include <stdio.h>
#include <omp.h>
#include <arm_neon.h>

static long long num_steps = 100000000000000000;
double step;

//cambiar esto
float32_t last;            // Sum of previous iteration for computing e
float32x4_t sum4;


int main(){

        step = 1.0 / (double)num_steps;
        omp_set_num_threads(4);

        float32_t cons[4] = {1,2,3,4};
        float32x4_t cons4 = vld1q_f32(cons);
        //double fact = 1;    
        float32_t fact[4] = {1,1,2,6};         // 0! = 1.  Keep running product of iteration numbers for factorial.
        float32x4_t fact4 = vld1q_f32(fact);
        float32_t invFact[4] = {1/fact4[0],1/fact4[1],1/fact4[2],1/fact4[3]};
        float32x4_t invFact4 = vld1q_f32(invFact);

        //unsigned int n = 0; 
        float32_t n[4] = {0,1,2,3};// Iteration.  Start with n=0;
        float32x4_t n4 = vld1q_f32(n);

        //double sum = 0;     
        float32_t sum[4] = {invFact[0],invFact[0]+invFact[1],invFact[0]+invFact[1]+invFact[2],invFact[0]+invFact[1]+invFact[2]+invFact[3]};         // Starting summation.  Keep a running sum of terms.
        float32x4_t sum4 = vld1q_f32(sum);


        #pragma omp parallel 
        {
        #pragma omp for reduction(+:last) private(sum4)
        
        for (int i = 1; i <= num_steps; i=i+4){
            last = (float32_t) vgetq_lane_f32(sum4, 3);
            //sum = sum + 1 / fact;
            float32_t invFact[4] = {1/fact4[0],1/fact4[1],1/fact4[2],1/fact4[3]};
            invFact4 = vld1q_f32(invFact);

            sum4 = vaddq_f32(sum4, invFact4);      // VADD.I16 d0,d0,d0
            //n = n+1;
            n4 = vaddq_f32(n4, cons4);           // VADD.I16 d0,d0,d08
            //fact = fact * n;
            //Vr[i] := Va[i] + Vb[i] * Vc[i]
            fact4 = vmulq_f32(fact4, n4);           // VMUL.I16 d0,d0,d0

        }
    
    }
    printf("resultado final de aproximacion: %f\n", last);


}

//https://www.wolframalpha.com/input/?i=sum&assumption=%7B%22F%22,+%22Sum%22,+%22sumfunction%22%7D+-%3E%221%2Fx!%22&assumption=%7B%22F%22,+%22Sum%22,+%22sumlowerlimit%22%7D+-%3E%220%22&assumption=%7B%22F%22,+%22Sum%22,+%22sumupperlimit2%22%7D+-%3E%22infinity%22&assumption=%7B%22FVarOpt%22,+%221%22%7D+-%3E+%7B%7B%7D,+%7B%7B%7B%22Sum%22,+%22sumvariable%22%7D%7D%7D,+%7B%7D%7D&assumption=%7B%22C%22,+%22sum%22%7D+-%3E+%7B%22Calculator%22,+%22dflt%22%7D
