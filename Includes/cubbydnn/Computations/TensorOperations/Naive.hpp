// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTETENSOR_HPP
#define CUBBYDNN_COMPUTETENSOR_HPP

#include <cassert>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <functional>

namespace CubbyDNN
{
class Naive
{
public:
    template <typename T>
    static void BasicLoop(T* destPtr, T* sourcePtr,
                          const std::function<T(T&)>& function,
                          std::size_t size)
    {
        for (std::size_t idx = 0; idx < size; ++idx)
        {
            destPtr[idx] = function(sourcePtr[idx]);
        }
    }

    template <typename T>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSizeA = inputShapeA.BatchSize();
        const auto batchSizeB = inputShapeB.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        if (batchSizeA != batchSizeB || batchSizeA != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");

        const auto matrixSizeA = inputShapeA.MatrixSize();
        const auto matrixSizeB = inputShapeB.MatrixSize();
        const auto outputMatrixSize = outputShape.MatrixSize();

        const auto rowSizeA = inputShapeA.Row();
        const auto colSizeA = inputShapeA.Col();

        const auto rowSizeB = inputShapeB.Row();
        const auto colSizeB = inputShapeB.Col();

        for (std::size_t batchIdx = 0; batchIdx < batchSizeA; ++batchIdx)
        {
            m_matAdd(
                static_cast<T*>(output.DataPtr) + outputMatrixSize * batchIdx,
                static_cast<T*>(inputA.DataPtr) + matrixSizeA * batchIdx,
                static_cast<T*>(inputB.DataPtr) + matrixSizeB * batchIdx,
                rowSizeA, colSizeA, rowSizeB, colSizeB);
        }
    }

    template <typename T>
    static void TensorMul(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSizeA = inputShapeA.BatchSize();
        const auto batchSizeB = inputShapeB.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        if (batchSizeA != batchSizeB || batchSizeA != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");

        const auto matrixSizeA = inputShapeA.MatrixSize();
        const auto matrixSizeB = inputShapeB.MatrixSize();
        const auto outputMatrixSize = outputShape.MatrixSize();

        const auto rowSizeA = inputShapeA.Row();
        const auto colSizeA = inputShapeA.Col();

        const auto rowSizeB = inputShapeB.Row();
        const auto colSizeB = inputShapeB.Col();

        for (std::size_t batchIdx = 0; batchIdx < batchSizeA; ++batchIdx)
        {
            m_matMul(
                static_cast<T*>(output.DataPtr) + outputMatrixSize * batchIdx,
                static_cast<T*>(inputA.DataPtr) + matrixSizeA * batchIdx,
                static_cast<T*>(inputB.DataPtr) + matrixSizeB * batchIdx,
                rowSizeA, colSizeA, rowSizeB, colSizeB);
        }
    }

    template <typename T>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShape.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        const auto rowSize = inputShape.Row();
        const auto colSize = inputShape.Col();

        if (batchSize != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");;
        for (std::size_t rowIdx = 0; rowIdx < inputShape.Row(); ++rowIdx)
        {
            for (std::size_t colIdx = 0; colIdx < inputShape.Col(); ++colIdx)
            {
                *(static_cast<T*>(output.DataPtr) + colIdx * colSize +
                  rowIdx) =
                    *(static_cast<T*>(input.DataPtr) + rowIdx * rowSize +
                      colIdx);
            }
        }
    }

    //! Transposes given matrix on sourcePtr and stores it in destPtr
    //! \tparam T : template param for type of data
    //! \param destPtr : ptr to transposed matrix allocated size must be
    //! (matrixSizeRow * matrixSizeCol)
    //! \param sourcePtr : ptr to matrix to transpose
    //! \param matrixSizeRow : row size of source matrix
    //! \param matrixSizeCol : column size of source matrix
    template <typename T>
    static void Transpose(T* destPtr, const T* sourcePtr,
                          std::size_t matrixSizeRow,
                          std::size_t matrixSizeCol) noexcept
    {
        constexpr std::size_t blockSize = 32;
        for (std::size_t rowIdx = 0; rowIdx * blockSize < matrixSizeRow;
             ++rowIdx)
        {
            for (std::size_t colIdx = 0; colIdx * blockSize < matrixSizeCol;
                 ++colIdx)
            {
                m_transPoseBlock<T>(destPtr, sourcePtr, matrixSizeRow,
                                    matrixSizeCol, blockSize, rowIdx, colIdx);
            }
        }
    }

private:
    template <typename T>
    static void m_matAdd(T* destPtr, const T* sourcePtrA, const T* sourcePtrB,
                         std::size_t rowSizeA, std::size_t colSizeA,
                         std::size_t rowSizeB, std::size_t colSizeB)
    {
        if (rowSizeA != rowSizeB || colSizeA != colSizeB)
            throw std::runtime_error(
                "Size of rows and columns should be identical");
        //! We can adjust this size
        const std::size_t blockSize = 32;

        for (std::size_t blockIdxRow = 0; blockIdxRow * blockSize < rowSizeA;
             ++blockIdxRow)
        {
            for (std::size_t blockIdxCol = 0;
                 blockIdxCol * blockSize < colSizeA; ++blockIdxCol)
            {
                for (std::size_t temp = 0; temp * blockSize < colSizeA; ++temp)
                {
                    m_blockedAdd<T>(destPtr, sourcePtrA, sourcePtrB, rowSizeA,
                                    colSizeA, blockIdxRow, blockIdxCol,
                                    blockSize);
                }
            }
        }
    }

    //! Multiplies two matrices (dest = A*B)
    //! \tparam T : type of the data to compute
    //! \param destPtr : pointer to destination
    //! \param sourcePtrA : pointer to source A
    //! \param sourcePtrB  : pointer to source B
    //! \param rowSizeA : size of rows of matrix A
    //! \param colSizeA : size of columns of matrix A
    //! \param rowSizeB : size of rows of matrix B
    //! \param colSizeB : size of columns of matrix B
    template <typename T>
    static void m_matMul(T* destPtr, const T* sourcePtrA, const T* sourcePtrB,
                         std::size_t rowSizeA, std::size_t colSizeA,
                         std::size_t rowSizeB, std::size_t colSizeB)
    {
        if (colSizeA != rowSizeB)
            throw std::runtime_error(
                "rowSizeA and colSizeB should be identical");

        constexpr std::size_t blockSize = 32;
        const auto destRowSize = rowSizeA;
        const auto destColSize = colSizeB;
        for (std::size_t blockIdxRow = 0; blockIdxRow * blockSize < rowSizeA;
             ++blockIdxRow)
        {
            //! TODO  : This can be done parallel
            for (std::size_t blockIdxCol = 0;
                 blockIdxCol * blockSize < colSizeB; ++blockIdxCol)
            {
                m_blockedMul<T>(destPtr, sourcePtrA, sourcePtrB, destRowSize,
                                destColSize, rowSizeB, blockIdxRow, blockIdxCol,
                                blockSize);
            }
        }
    }

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
                                        std::size_t matrixSizeRow,
                                        std::size_t matrixSizeCol,
                                        std::size_t blockSize,
                                        std::size_t blockIdxRow,
                                        std::size_t blockIdxCol) noexcept
    {
        const auto rowSize = (matrixSizeRow < (blockIdxRow + 1) * blockSize)
                                 ? matrixSizeRow - blockIdxRow * blockSize
                                 : blockSize;

        const auto colSize = (matrixSizeCol < (blockIdxCol + 1) * blockSize)
                                 ? matrixSizeCol - blockIdxCol * blockSize
                                 : blockSize;

        for (std::size_t blockRowIdx = 0; blockRowIdx < rowSize; ++blockRowIdx)
        {
            for (std::size_t blockColIdx = 0; blockColIdx < colSize;
                 ++blockColIdx)
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
    static void m_blockedAdd(T* destPtr, const T* sourcePtrA,
                             const T* sourcePtrB, std::size_t destRowSize,
                             std::size_t destColSize, std::size_t blockIdxRow,
                             std::size_t blockIdxCol, std::size_t blockSize)
    {
        const auto rowSize = (destRowSize < (blockIdxRow + 1) * blockSize)
                                 ? destRowSize - blockIdxRow * blockSize
                                 : blockSize;

        const auto colSize = (destColSize < (blockIdxCol + 1) * blockSize)
                                 ? destColSize - blockIdxCol * blockSize
                                 : blockSize;

        for (std::size_t blockRowIdx = 0; blockRowIdx < rowSize; ++blockRowIdx)
        {
            for (std::size_t blockColIdx = 0; blockColIdx < colSize;
                 ++blockColIdx)
            {
                *(destPtr + blockSize * blockRowIdx + blockColIdx) =
                    *(sourcePtrA + blockSize * blockRowIdx + blockColIdx) +
                    *(sourcePtrB + blockSize * blockRowIdx + blockColIdx);
            }
        }
    }

    //! Performs multiplication of single block in blocked matrix multiplication
    //! (dest = A*B) \tparam T : type of the data used for calculation \param
    //! destPtr : pointer to rowMajor destination matrix \param sourcePtrA :
    //! pointer to rowMajor source matrix A \param sourcePtrB : pointer to
    //! rowMajor source matrix B \param destRowSize :  Total size of rows in
    //! destination matrix \param destColSize  : Total size of columns in
    //! destination matrix \param rowSizeB : Total size of row of matrix B (or
    //! size of columns of matrixA) \param blockIdxRow : Block index of the row
    //! of destination matrix \param blockIdxCol : Block index of the column of
    //! destination matrix \param blockSize : Size of the block
    template <typename T>
    static void m_blockedMul(T* destPtr, const T* sourcePtrA,
                             const T* sourcePtrB, std::size_t destRowSize,
                             std::size_t destColSize, std::size_t rowSizeB,
                             std::size_t blockIdxRow, std::size_t blockIdxCol,
                             std::size_t blockSize) noexcept
    {
        const auto tempBlockSize = rowSizeB / blockSize + 1;

        for (std::size_t tempBlockIdx = 0; tempBlockIdx < tempBlockSize;
             ++tempBlockIdx)
        {
            const auto tempSize = (rowSizeB < (tempBlockIdx + 1) * blockSize)
                                      ? rowSizeB - tempBlockIdx * blockSize
                                      : blockSize;
            const auto rowSize = (destRowSize < (blockIdxRow + 1) * blockSize)
                                     ? destRowSize - blockIdxRow * blockSize
                                     : blockSize;
            const auto colSize = (destColSize < (blockIdxCol + 1) * blockSize)
                                     ? destColSize - blockIdxCol * blockSize
                                     : blockSize;

            for (std::size_t rowIdx = 0; rowIdx < rowSize; ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    T* dest =
                    (destPtr +
                     (blockIdxRow * blockSize + rowIdx) * destRowSize +
                     (blockIdxCol * blockSize + colIdx));
                    for (std::size_t tempIdx = 0; tempIdx < tempSize; ++tempIdx)
                    {
                        *dest +=
                            *(sourcePtrA +
                              (blockIdxRow * blockSize + rowIdx) * destRowSize +
                              (tempBlockIdx * blockSize + tempIdx)) *
                            *(sourcePtrB +
                              (tempBlockIdx * blockSize + tempIdx) * rowSizeB +
                              (blockIdxCol * blockSize + colIdx));
                    }
                }
        }
    }

    template <typename T>
    static constexpr T m_pow(T num, std::size_t pow)
    {
        return (pow >= sizeof(unsigned int) * 8)
                   ? 0
                   : pow == 0
                   ? 1
                   : num * m_pow(num, pow - 1);
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTETENSOR_HPP
