// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cmath>
#include <cubbydnn/Utils/Shape.hpp>
#include <random>

namespace CubbyDNN
{
template <typename T>
void LecunNormal(const Shape& shape, T* data)
{
    std::random_device rd;
    std::mt19937 engine(rd());
    const auto fanIn = shape.Row();
    const auto variance = 1 / sqrt(static_cast<T>(fanIn));
    std::normal_distribution<T> normal(0, variance);

    const auto matrixSize = shape.PaddedMatrixSize();
    const auto batchSize = shape.BatchSize();
    const auto colSize = (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

    const auto rowMean = shape.Row() / static_cast<T>(2);
    const auto colMean = shape.Col() / static_cast<T>(2);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
            {
                if (colIdx < shape.Row())
                    *(data + batchIdx * matrixSize + rowIdx * colSize +
                      colIdx) = normal(engine);
            }
}

template <typename T>
void XavierNormal(const Shape& shape, T* data)
{
    std::random_device rd;
    std::mt19937 engine(rd());

    const auto fanIn = shape.Row();
    const auto fanOut = shape.Col();
    const auto variance = 1 / sqrt(static_cast<T>(fanIn + fanOut));

    std::normal_distribution<T> normal(0, variance);
    const auto matrixSize = shape.PaddedMatrixSize();
    const auto batchSize = shape.BatchSize();
    const auto colSize = (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

    const auto rowMean = shape.Row() / static_cast<T>(2);
    const auto colMean = shape.Col() / static_cast<T>(2);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
            {
                if (colIdx < shape.Row())
                    *(data + batchIdx * matrixSize + rowIdx * colSize +
                      colIdx) = normal(engine);
            }
}

template <typename T>
void HeNormal(const Shape& shape, T* data)
{
    std::random_device rd;
    std::mt19937 engine(rd());

    const auto fanIn = shape.Row();
    const auto variance = 2 / sqrt(static_cast<T>(fanIn));

    std::normal_distribution<T> normal(0, variance);
    const auto matrixSize = shape.PaddedMatrixSize();
    const auto batchSize = shape.BatchSize();
    const auto colSize = (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

    const auto rowMean = shape.Row() / static_cast<T>(2);
    const auto colMean = shape.Col() / static_cast<T>(2);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
            {
                if (colIdx < shape.Row())
                    *(data + batchIdx * matrixSize + rowIdx * colSize +
                      colIdx) = normal(engine);
            }
}

template <typename T>
void Zeros(const Shape& shape, T* data)
{
    const auto matrixSize = shape.PaddedMatrixSize();
    const auto batchSize = shape.BatchSize();
    const auto colSize = (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

    const auto rowMean = shape.Row() / static_cast<T>(2);
    const auto colMean = shape.Col() / static_cast<T>(2);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
            {
                if (colIdx < shape.Row())
                    *(data + batchIdx * matrixSize + rowIdx * colSize +
                      colIdx) = static_cast<T>(0);
            }
}

template <typename T>
void Ones(const Shape& shape, T* data)
{
    const auto matrixSize = shape.PaddedMatrixSize();
    const auto batchSize = shape.BatchSize();
    const auto colSize = (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

    const auto rowMean = shape.Row() / static_cast<T>(2);
    const auto colMean = shape.Col() / static_cast<T>(2);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
            {
                if (colIdx < shape.Row())
                    *(data + batchIdx * matrixSize + rowIdx * colSize +
                      colIdx) = static_cast<T>(1);
            }
}
} // namespace CubbyDNN
