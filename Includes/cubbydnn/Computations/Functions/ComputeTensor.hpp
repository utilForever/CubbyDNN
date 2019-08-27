// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTETENSOR_HPP
#define CUBBYDNN_COMPUTETENSOR_HPP

#include <cstdio>
#include <functional>
#include <iostream>

namespace CubbyDNN
{
class ComputeTensor
{
 public:
    template <typename T>
    static void BasicLoop(T* destPtr, T* sourcePtr,
                          const std::function<T(T&)>& function, size_t size)
    {
#ifdef USE_OPENMP
#pragma omp parallel default(none)
#endif
        for (size_t idx = 0; idx < size; ++idx)
        {
            destPtr[idx] = function(sourcePtr[idx]);
        }
    }

    template <typename T>
    static void NaiveTranspose(T* destPtr, T* sourcePtr, size_t matrixSizeRow,
                               size_t matrixSizeCol)
    {
        for (size_t rowIdx = 0; rowIdx < matrixSizeRow; ++rowIdx)
        {
            for (size_t colIdx = 0; colIdx < matrixSizeCol; ++colIdx)
            {
                *(destPtr + colIdx * matrixSizeCol + rowIdx) =
                    *(sourcePtr + rowIdx * matrixSizeRow + colIdx);
            }
        }
    }

    /**
     * Transposes given matrix on sourcePtr and stores it in destPtr
     * @tparam T : template param for type of data
     * @param destPtr : ptr to transposed matrix allocated size must be
     * (matrixSizeRow * matrixSizeCol)
     * @param sourcePtr : ptr to matrix to transpose
     * @param matrixSizeRow : row size of source matrix
     * @param matrixSizeCol : column size of source matrix
     */
    template <typename T>
    static void Transpose(T* destPtr, T* sourcePtr, size_t matrixSizeRow,
                          size_t matrixSizeCol)
    {
        /// optimized to intel skylake architecture
        constexpr size_t blockSize = m_pow(2, 10) / sizeof(T);
        for (size_t rowIdx = 0; rowIdx < matrixSizeRow; rowIdx += blockSize)
        {
            for (size_t colIdx = 0; colIdx < matrixSizeCol; colIdx += blockSize)
            {
                m_transPoseBlock<T>(destPtr, sourcePtr, matrixSizeRow,
                                    matrixSizeCol, blockSize, rowIdx, colIdx);
            }
        }
    }

 private:
    /**
     *
     * @tparam T
     * @param destPtr
     * @param sourcePtr
     * @param matrixSizeRow
     * @param matrixSizeCol
     * @param blockSize
     * @param blockIdxRow
     * @param blockIdxCol
     */
    template <typename T>
    static void m_transPoseBlock(T* destPtr, T* sourcePtr, size_t matrixSizeRow,
                                 size_t matrixSizeCol, size_t blockSize,
                                 size_t blockIdxRow, size_t blockIdxCol)
    {
        size_t rowSize = (matrixSizeRow < (blockIdxRow + 1) * blockSize)
                             ? matrixSizeRow - blockIdxRow * blockSize
                             : blockSize;

        size_t colSize = (matrixSizeCol < (blockIdxCol + 1) * blockSize)
                             ? matrixSizeCol - blockIdxCol * blockSize
                             : blockSize;

        for (size_t blockRowIdx = 0; blockRowIdx < rowSize; ++blockRowIdx)
        {
            for (size_t blockColIdx = 0; blockColIdx < colSize; ++blockColIdx)
            {
                *(destPtr + blockIdxCol * matrixSizeCol * blockSize +
                  matrixSizeCol * blockColIdx + blockIdxRow * blockSize +
                  blockRowIdx) =
                    *(sourcePtr + blockIdxRow * matrixSizeRow * blockSize +
                      matrixSizeRow * blockRowIdx + blockIdxCol * blockSize +
                      blockColIdx);
            }
        }
    }

    template <typename T>
    static constexpr T m_pow(T num, size_t pow)
    {
        return (pow >= sizeof(unsigned int) * 8)
                   ? 0
                   : pow == 0 ? 1 : num * m_pow(num, pow - 1);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTETENSOR_HPP