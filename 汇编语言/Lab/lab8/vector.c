#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdint.h"

#define ARRAY_SIZE 1024 * 1024 * 16
#define TIMES 100

uint32_t src0[ARRAY_SIZE];
uint32_t src1[ARRAY_SIZE];
uint32_t dest[ARRAY_SIZE];

#define COMPUTE_KERNEL() \
do \
{ \
    uint32_t temp;\
    temp = src0[i] * 0x12345678;\
    temp += src1[i] * 0x76543210;\
    temp *= 0xA0A00505;\
    dest[i] = temp + src1[i];\
} \
while (0)

uint64_t rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc": "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

uint32_t checksum(void)
{
    uint32_t final = 0;
    for (int i = 0; i < ARRAY_SIZE; ++i)
        final += dest[i];
    return final;
}

void __attribute__((optimize("O0"))) raw_calc_naive(void)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
        COMPUTE_KERNEL();
}

void __attribute__((optimize("O2"))) raw_calc_expert(void)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
        COMPUTE_KERNEL();
}

void __attribute__((optimize("O3"))) raw_calc_sse(void)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
        COMPUTE_KERNEL();
}

void __attribute__((optimize("O3"), __target__("arch=core-avx2"))) raw_calc_avx_auto(void)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
        COMPUTE_KERNEL();
}

void __attribute__((optimize("O3"), __target__("arch=core-avx2"))) raw_calc_avx_manual(void)
{
    const uint32_t const0 = 0x12345678;
    const uint32_t const1 = 0x76543210;
    const uint32_t const2 = 0xA0A00505;
    const uint64_t limit = ARRAY_SIZE * 4;
    __asm__ __volatile__
    (
        "xor            %%rcx,              %%rcx\n"
        "mov            $0x20,              %%rax\n"
        "mov            $0x40,              %%rdx\n"
        "mov            $0x60,              %%rbp\n"
        "lea            %[src0],            %%rbx\n"
        "lea            %[src1],            %%rsi\n"
        "lea            %[dest],            %%rdi\n"
        "vpbroadcastd   %[const0],          %%ymm13\n"
        "vpbroadcastd   %[const1],          %%ymm14\n"
        "vpbroadcastd   %[const2],          %%ymm15\n"
        "1:\n"

        "vpmulld        (%%rbx,%%rcx,1),    %%ymm13,        %%ymm0\n"
        "vpmulld        (%%rbx,%%rax,1),    %%ymm13,        %%ymm3\n"
        "vpmulld        (%%rbx,%%rdx,1),    %%ymm13,        %%ymm6\n"
        "vpmulld        (%%rbx,%%rbp,1),    %%ymm13,        %%ymm9\n"

        "vmovdqu        (%%rsi,%%rcx,1),    %%ymm1\n"
        "vpmulld        %%ymm1,             %%ymm14,        %%ymm2\n"
        "vmovdqu        (%%rsi,%%rax,1),    %%ymm4\n"
        "vpmulld        %%ymm4,             %%ymm14,        %%ymm5\n"
        "vmovdqu        (%%rsi,%%rdx,1),    %%ymm7\n"
        "vpmulld        %%ymm7,             %%ymm14,        %%ymm8\n"
        "vmovdqu        (%%rsi,%%rbp,1),    %%ymm10\n"
        "vpmulld        %%ymm10,            %%ymm14,        %%ymm11\n"

        "vpaddd         %%ymm0,             %%ymm2,         %%ymm2\n"
        "vpaddd         %%ymm3,             %%ymm5,         %%ymm5\n"
        "vpaddd         %%ymm6,             %%ymm8,         %%ymm8\n"
        "vpaddd         %%ymm9,             %%ymm11,        %%ymm11\n"

        "vpmulld        %%ymm2,             %%ymm15,        %%ymm2\n"
        "vpmulld        %%ymm5,             %%ymm15,        %%ymm5\n"
        "vpmulld        %%ymm8,             %%ymm15,        %%ymm8\n"
        "vpmulld        %%ymm11,            %%ymm15,        %%ymm11\n"

        "vpaddd         %%ymm2,             %%ymm1,         %%ymm2\n"
        "vmovdqu        %%ymm2,             (%%rdi,%%rcx,1)\n"
        "vpaddd         %%ymm5,             %%ymm4,         %%ymm5\n"
        "vmovdqu        %%ymm5,             (%%rdi,%%rax,1)\n"
        "vpaddd         %%ymm8,             %%ymm7,         %%ymm8\n"
        "vmovdqu        %%ymm8,             (%%rdi,%%rdx,1)\n"
        "vpaddd         %%ymm11,            %%ymm10,        %%ymm11\n"
        "vmovdqu        %%ymm11,            (%%rdi,%%rbp,1)\n"

        "add            $0x80,              %%rcx\n"
        "add            $0x80,              %%rax\n"
        "add            $0x80,              %%rdx\n"
        "add            $0x80,              %%rbp\n"

        "cmp            %%rcx,              %[limit]\n"
        "jne            1b\n"
        : [dest] "=m"(dest)
        : [src0] "m"(src0), [src1] "m"(src1), [const0] "m"(const0), [const1] "m"(const1), [const2] "m"(const2), [limit] "r"(limit)
        : "rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%rbp", "memory", "cc"
    );
}

void __attribute__((optimize("O3"),__target__("arch=icelake-server"))) raw_calc_avx512(void)
{
    const uint32_t const0 = 0x12345678;
    const uint32_t const1 = 0x76543210;
    const uint32_t const2 = 0xA0A00505;
    const uint64_t limit = ARRAY_SIZE * 4;

    __asm__ __volatile__
    (
        "xor            %%rcx,              %%rcx\n"
        "lea            %[src0],            %%rbx\n"
        "lea            %[src1],            %%rsi\n"
        "lea            %[dest],            %%rdi\n"
        "vpbroadcastd   %[const0],          %%zmm13\n"
        "vpbroadcastd   %[const1],          %%zmm14\n"
        "vpbroadcastd   %[const2],          %%zmm15\n"
        "1:\n"
        "vpmulld        (%%rbx,%%rcx,1),    %%zmm13,        %%zmm0\n"
        "vmovdqu32      (%%rsi,%%rcx,1),    %%zmm1\n"
        "vpmulld        %%zmm1,             %%zmm14,        %%zmm2\n"
        "vpaddd         %%zmm0,             %%zmm2,         %%zmm2\n"
        "vpmulld        %%zmm2,             %%zmm15,        %%zmm2\n"
        "vpaddd         %%zmm2,             %%zmm1,         %%zmm2\n"
        "vmovdqu32      %%zmm2,             (%%rdi,%%rcx,1)\n"
        "add            $0x40,              %%rcx\n"
        "cmp            %%rcx,              %[limit]\n"
        "jne            1b\n"
        :[dest]"=m"(dest)
        :[src0] "m"(src0), [src1] "m"(src1), [const0] "m"(const0), [const1] "m"(const1), [const2] "m"(const2), [limit] "r"(limit)
        :"%rbx","%rcx","%rsi","%rdi","memory","cc"
    );
}

void test(char* opt, void (*func)(void))
{
    uint64_t start;
    uint64_t end;

    memset(dest, 0x00, ARRAY_SIZE*sizeof(uint32_t));

    start = rdtsc();
    for(int i = 0 ; i < TIMES; i++)
        func();
    end = rdtsc();

    printf("%s - %lu - %llu cycles\n",opt, checksum(), end-start);
}

int main(void)
{
    memset(src0, 0x12, ARRAY_SIZE * sizeof(uint32_t));
    memset(src1, 0x34, ARRAY_SIZE * sizeof(uint32_t));
    raw_calc_naive();

    // test("avx-512",raw_calc_avx512);
    test("naive",raw_calc_naive);
    test("expert",raw_calc_expert);
    test("sse",raw_calc_sse);
    test("avx-auto",raw_calc_avx_auto);
    test("avx-manual",raw_calc_avx_manual);
    return 0;
}