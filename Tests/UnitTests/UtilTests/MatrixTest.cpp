//
// Created by jwkim98 on 8/25/19.
//

#include "MatrixTest.hpp"

namespace UtilTest
{
using namespace CubbyDNN;
void MatrixTransposeTest()
{
    {
        const size_t rowSize = 150;
        const size_t colSize = 150;
        float* sourcePtr = CreateMatrix<float>(rowSize, colSize);
        float* destPtr = CreateMatrix<float>(colSize, rowSize);
        float* checkPtr = CreateMatrix<float>(colSize, rowSize);

        auto t1 = std::chrono::high_resolution_clock::now();
        ComputeTensor::NaiveTranspose<float>(checkPtr, sourcePtr, rowSize,
                                             colSize);
        auto t2 = std::chrono::high_resolution_clock::now();
        ComputeTensor::Transpose<float>(destPtr, checkPtr, colSize, rowSize);
        auto t3 = std::chrono::high_resolution_clock::now();

        /// Check if transpose was valid
        for (size_t count = 0; count < rowSize * colSize; ++count)
        {
            assert(*(sourcePtr + count) == *(destPtr + count));
        }

        std::cout << "Naive transpose took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t2 -
                                                                           t1)
                         .count()
                  << std::endl;

        std::cout << "Optimized transpose took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t3 -
                                                                           t2)
                         .count()
                  << std::endl;
        //        free(sourcePtr);
        //        free(destPtr);
        //        free(checkPtr);
        delete[] sourcePtr;
        delete[] destPtr;
        delete[] checkPtr;
    }
}

void MatrixMultiplicationTest()
{
    const size_t rowSize = 100;
    const size_t colSize = 100;
    float* sourcePtr1 = CreateMatrix<float>(rowSize, colSize);
    float* sourcePtr2 = CreateMatrix<float>(colSize, rowSize);
    float* destPtr1 = CreateMatrix<float>(colSize, rowSize, true);
    float* destPtr2 = CreateMatrix<float>(colSize, rowSize, true);


    auto t1 = std::chrono::high_resolution_clock::now();
    ComputeTensor::NaiveMatmul<float>(destPtr1, sourcePtr1, sourcePtr2, rowSize,
                                      colSize, colSize, rowSize);
    auto t2 = std::chrono::high_resolution_clock::now();
    ComputeTensor::Matmul<float>(destPtr2, sourcePtr1, sourcePtr2, rowSize,
                                 colSize, colSize, rowSize);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "Naive matmul took "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count()
              << std::endl;

    std::cout << "Optimized matmul took "
              << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2)
                     .count()
              << std::endl;

    for (size_t count = 0; count < rowSize * colSize; ++count)
    {
        assert(*(destPtr1 + count) == *(destPtr2 + count));
//        std::cout << *(destPtr1 + count) << " | " << *(destPtr2 + count)
//                  << std::endl;
    }

    delete[] sourcePtr1;
    delete[] sourcePtr2;
    delete[] destPtr1;
    delete[] destPtr2;
}

TEST(MatrixTransposeTest, MatrixTranspose)
{
    MatrixTransposeTest();
}

TEST(MatrixMultiplicationTest, MatrixMultiplication)
{
    MatrixMultiplicationTest();
}

}  // namespace UtilTest