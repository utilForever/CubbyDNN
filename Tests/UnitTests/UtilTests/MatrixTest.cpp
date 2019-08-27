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

TEST(MatrixTranspose_test, MatrixTranspose)
{
    MatrixTransposeTest();
}

}  // namespace UtilTest
