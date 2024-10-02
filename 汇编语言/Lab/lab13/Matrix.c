#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdint.h"

#define ARRAY_SIZE 1024
#define TIMES 100

uint32_t src0[ARRAY_SIZE][ARRAY_SIZE];
uint32_t src1[ARRAY_SIZE][ARRAY_SIZE];
uint32_t dest[ARRAY_SIZE][ARRAY_SIZE];
uint64_t time[6];
uint64_t sum[6];

#define COMPUTE_KERNEL() \
do \
{ \
    for(int i = 0; i < 1024; ++i)\
    {\
        for(int j = 0; j < 1024; ++j)\
        {\
            for(int k = 0; k < 1024; ++k)\
            {\
                dest[i][j] += src0[i][k] * src1[j][k];\
            }\
        } \
    }\
} \
while (0)


void readB()
{
    char path[100] = "H:\\data1\\B.txt";
    freopen(path,"r",stdin);
    for(int i = 0; i < 1024; ++i)
    {
        for(int j = 0; j < 1024; ++j)
        {
            scanf("%d",&src1[j][i]);
        }
    }
}

void readA(int num)
{
    if(num < 10)
    {
        char txt[6] = "1.txt";
        char path1[100] = "H:\\data1\\A";
        txt[0] = num;
        txt[0] += 48;
        strncat(path1,txt,5);
        freopen(path1,"r",stdin);
        for(int i=0;i<1024;++i)
        {
            for(int j=0;j<1024;++j)
            {
                scanf("%d",&src0[i][j]);
            }
        }
    }
    else
    {
        char txt1[7] = "11.txt";
        char path1[100] = "H:\\data1\\A";
        txt1[1] = num-10;
        txt1[1] += 48;
        strncat(path1,txt1,6);
        freopen(path1,"r",stdin);
        for(int i = 0; i < 1024; ++i)
        {
            for(int j = 0; j < 1024; ++j)
            {
                scanf("%d",&src0[i][j]);
            }
        }
    }
}


uint64_t rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc": "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

uint64_t checksum(void)
{
    uint64_t final = 0;
    for(int i=0;i<1024;++i)
        {
            for(int j=0;j<1024;++j)
            {
                final += dest[i][j];
            }
        }
    return final;
}

void __attribute__((optimize("O0"))) raw_calc_naive(void)
{
    COMPUTE_KERNEL();
}

void __attribute__((optimize("O2"))) raw_calc_expert(void)
{
    COMPUTE_KERNEL();
}

void __attribute__((optimize("O3"))) raw_calc_sse(void)
{
    COMPUTE_KERNEL();
}
void __attribute__((optimize("O3"), __target__("arch=core-avx2"))) raw_calc_avx_auto(void)
{
    COMPUTE_KERNEL();
}

void __attribute__((optimize("03"),__target__("arch=core-avx2"))) raw_calc_avx_manual(void)
{
    const uint64_t line=ARRAY_SIZE*4;
    const uint64_t matrix=ARRAY_SIZE*ARRAY_SIZE*4;
    const uint32_t zero=0x0;           
    const uint32_t temp[8];
    __asm__ __volatile__
    (
        "xor                %%rcx,              %%rcx\n"                    //src0
        "xor                %%r8,               %%r8\n"                     //src1
        "xor                %%r9,               %%r9\n"                     //temp
        "xor                %%r10,              %%r10\n"                    //judge
        "xor                %%r11,              %%r11\n"                    //dest
        "xor                %%eax,              %%eax\n"                    //sum
        "lea                %[src0],            %%rbx\n"
        "lea                %[src1],            %%rsi\n"
        "lea                %[dest],            %%rdi\n"
        "lea                %[temp],            %%rdx\n"
        
        "1:\n"
        "vpbroadcastd       %[zero],            %%ymm0\n"                       
        
        "2:\n"
        "vmovdqu            (%%rbx,%%rcx,1),    %%ymm3\n"
        "vmovdqu            (%%rsi,%%r8,1),     %%ymm6\n"
        "vpmulld            %%ymm3,             %%ymm6,             %%ymm9\n"
        "vpaddd             %%ymm9,             %%ymm0,             %%ymm0\n"
        "add                $0x20,              %%rcx\n"
        "add                $0x20,              %%r8\n"
        "add                $0x20,              %%r10\n"
        "cmp                %%r10,              %[line]\n"                 //if this line end
        "jne                2b\n"                                          //not end
        
        "xor                %%r10,              %%r10\n"                   //line end
        "xor                %%eax,              %%eax\n"                   //get sum to zeor
        "vmovdqu            %%ymm0,             (%%rdx,%%r9,1)\n"
        
        "push               %%rcx\n"                                       //get sum
        "mov                $0x08,              %%rcx\n"
        
        "suming:\n"
        "add                (%%rdx,%%r9,1),     %%eax\n"
        "add                $0x04,              %%r9\n"
        "loop               suming\n"
        
        "xor                %%r9,               %%r9\n"
        "pop                %%rcx\n"
        
        "mov                %%eax,              (%%rdi,%%r11,1)\n"          //sum is in eax,move it to dest
        "add                $0x04,              %%r11\n"                    //next dest
        "cmp                %%r11,              %[matrix]\n"                //if dest end
        "je                 ending\n"                                       //ending is end
        "cmp                %%r8,               %[matrix]\n"                //if src1 all end
        "je                 nextsrc0\n"                                     //all end, next src0 line
        "sub                %[line],            %%rcx\n"                    //back this line of src0
        "jmp                1b\n"
        
        "nextsrc0:\n"
        "xor                %%r8,               %%r8\n"
        "jmp                1b\n"
        
        "ending:\n"
        :[dest]"=m"(dest)
        :[zero]"m"(zero),[src0]"m"(src0),[src1]"m"(src1),[temp]"m"(temp),[line]"m"(line),[matrix]"r"(matrix)
        :"%rax","%rbx","%rcx","%rsi","%rdi","%rdx","%r8","r9","%r10","%r11","memory","cc"
    );
}

void calcu(int x,char* opt, void (*func)(void))
{
    memset(dest, 0x00, ARRAY_SIZE*ARRAY_SIZE*sizeof(uint32_t));
    uint64_t start,end;
    start = rdtsc();
    func();
    end = rdtsc();
    time[x] += (end - start);
    sum[x] += checksum();
}

int main(void)
{
    readB();
    for(int i=1;i<=16;++i)
    {
        readA(i);
        calcu(1,"naive",raw_calc_naive);
        calcu(2,"expert",raw_calc_expert);
        calcu(3,"sse",raw_calc_sse);
        calcu(4,"avx-auto",raw_calc_avx_auto);
        calcu(5,"avx-manual",raw_calc_avx_manual);
        printf("Matrix %d finish\n",i);
    }
    printf("naive - %llu - %llu cycles\n",sum[1],time[1]);
    printf("expert - %llu - %llu cycles\n",sum[2],time[2]);
    printf("sse - %llu - %llu cycles\n",sum[3],time[3]);
    printf("avx-auto - %llu - %llu cycles\n",sum[4],time[4]);
    printf("avx-manual - %llu - %llu cycles\n",sum[5],time[5]);
    return 0;
}
