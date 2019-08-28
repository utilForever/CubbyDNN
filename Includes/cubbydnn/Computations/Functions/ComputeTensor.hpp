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
    static void Matmul(T* destPtr, const T* sourceA, const T* sourceB,
                       size_t rowSizeA, size_t colSizeA, size_t rowSizeB,
                       size_t colSizeB)
    {
        assert(colSizeA == rowSizeB);
        constexpr size_t blockSize = 32;
        for (size_t rowIdx = 0; rowIdx * blockSize < rowSizeA; ++rowIdx)
        {
            for (size_t colIdx = 0; colIdx * blockSize < colSizeB; ++colIdx)
            {
                for (size_t temp = 0; temp * blockSize < colSizeA; ++temp)
                {
                    m_matMulBlock<T>(destPtr, sourceA, sourceB, rowSizeA,
                                     colSizeA, rowSizeB, colSizeB, rowIdx,
                                     colIdx, temp, blockSize);
                }
            }
        }
    }

    template <typename T>
    static void NaiveMatmul(T* destPtr, const T* sourceA, const T* sourceB,
                            size_t rowSizeA, size_t colSizeA, size_t rowSizeB,
                            size_t colSizeB)
    {
        assert(colSizeA == rowSizeB);
        for (size_t rowIdx = 0; rowIdx < rowSizeA; ++rowIdx)
        {
            for (size_t colIdx = 0; colIdx < colSizeB; ++colIdx)
            {
                for (size_t tempIdx = 0; tempIdx < colSizeA; ++tempIdx)
                {
                    *(destPtr + rowIdx * rowSizeA + colIdx) +=
                        *(sourceA + rowIdx * rowSizeA + tempIdx) *
                        *(sourceB + tempIdx * rowSizeB + colIdx);
                }
            }
        }
    }

    template <typename T>
    static void NaiveTranspose(T* destPtr, const T* sourcePtr,
                               size_t matrixSizeRow,
                               size_t matrixSizeCol) noexcept
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
    static void Transpose(T* destPtr, const T* sourcePtr, size_t matrixSizeRow,
                          size_t matrixSizeCol) noexcept
    {
        /// optimized to intel skylake architecture
        constexpr size_t blockSize = 32;
        for (size_t rowIdx = 0; rowIdx * blockSize < matrixSizeRow; ++rowIdx)
        {
            for (size_t colIdx = 0; colIdx * blockSize < matrixSizeCol;
                 ++colIdx)
            {
                m_transPoseBlock<T>(destPtr, sourcePtr, matrixSizeRow,
                                    matrixSizeCol, blockSize, rowIdx, colIdx);
            }
        }
    }

    /**
     *
     * @tparam T
     * @param dst
     * @param src
     * @param n
     * @param p
     */
    template <typename T>
    static void Transpose2(T* dst, const T* src, size_t n, size_t p) noexcept
    {
        size_t block = 32;
        for (size_t i = 0; i < n; i += block)
        {
            for (size_t j = 0; j < p; ++j)
            {
                for (size_t b = 0; b < block && i + b < n; ++b)
                {
                    dst[j * n + i + b] = src[(i + b) * p + j];
                }
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
    static void inline m_transPoseBlock(T* destPtr, const T* sourcePtr,
                                        size_t matrixSizeRow,
                                        size_t matrixSizeCol, size_t blockSize,
                                        size_t blockIdxRow,
                                        size_t blockIdxCol) noexcept
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
                *(destPtr +
                  (blockIdxCol * blockSize + blockColIdx) * matrixSizeCol +
                  blockIdxRow * blockSize + blockRowIdx) =
                    *(sourcePtr +
                      (blockIdxRow * blockSize + blockRowIdx) * matrixSizeRow +
                      blockIdxCol * blockSize + blockColIdx);
            }
        }
    }

    template <typename T>
    static void inline m_matMulBlock(T* destPtr, const T* sourcePtr1,
                                     const T* sourcePtr2, size_t rowSizeA,
                                     size_t colSizeA, size_t rowSizeB,
                                     size_t colSizeB, size_t blockIdxRow,
                                     size_t blockIdxCol, size_t blockIdxTemp,
                                     size_t blockSize) noexcept
    {
        size_t rowSize = (rowSizeA < (blockIdxRow + 1) * blockSize)
                             ? rowSizeA - blockIdxRow * blockSize
                             : blockSize;

        size_t colSize = (colSizeB < (blockIdxCol + 1) * blockSize)
                             ? colSizeB - blockIdxCol * blockSize
                             : blockSize;

        size_t tempSize = (colSizeA < (blockIdxTemp + 1) * blockSize)
                              ? colSizeA - blockIdxTemp * blockSize
                              : blockSize;

        for (size_t blockRowIdx = 0; blockRowIdx < rowSize; ++blockRowIdx)
        {
            for (size_t blockColIdx = 0; blockColIdx < colSize; ++blockColIdx)
            {
                T* dest = (destPtr +
                           (blockIdxRow * blockSize + blockRowIdx) * rowSizeA +
                           (blockIdxCol * blockSize + blockColIdx));
                for (size_t tempIdx = 0; tempIdx < tempSize; ++tempIdx)
                {
                    *dest +=
                        *(sourcePtr1 +
                          (blockIdxRow * blockSize + blockRowIdx) * rowSizeA +
                          (blockIdxTemp * blockSize + tempIdx)) *
                        *(sourcePtr2 +
                          (blockIdxTemp * blockSize + tempIdx) * rowSizeB +
                          (blockIdxCol * blockSize + blockColIdx));
                }
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