// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_INITIALIZEROP_HPP
#define CUBBYDNN_INITIALIZEROP_HPP

#include <cmath>
#include <cubbydnn/Utils/Shape.hpp>
#include <random>

namespace CubbyDNN
{
class InitializerOperations
{
public:
    template <typename T>
    static void RandomNormal(const Shape& shape, T mean, T stddev, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::normal_distribution<T> normal(mean, stddev);

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
    static void RandomUniform(const Shape& shape, T min, T max, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

        const auto rowMean = shape.Row() / static_cast<T>(2);
        const auto colMean = shape.Col() / static_cast<T>(2);

        if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> uniform(min, max);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(min, max);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void LecunNormal(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto fanIn = shape.Row();
        const auto stddev = static_cast<T>(1 / sqrt(static_cast<T>(fanIn)));
        std::normal_distribution<T> normal(0, stddev);

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
    static void LecunUniform(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto fanIn = shape.Row();
        const auto range = static_cast<T>(sqrt(3 / static_cast<T>(fanIn)));

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

        const auto rowMean = shape.Row() / static_cast<T>(2);
        const auto colMean = shape.Col() / static_cast<T>(2);

        if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void XavierNormal(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.Row();
        const auto fanOut = shape.Col();
        const auto stddev = static_cast<T>(sqrt(
            2 / static_cast<T>(fanIn + fanOut)));

        std::normal_distribution<T> normal(0, stddev);
        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
    static void XavierUniform(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.Row();
        const auto fanOut = shape.Col();
        const auto range =
            static_cast<T>(sqrt(6 / static_cast<T>(fanIn + fanOut)));

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

        const auto rowMean = shape.Row() / static_cast<T>(2);
        const auto colMean = shape.Col() / static_cast<T>(2);

        if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void HeNormal(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.Row();
        const auto stddev = static_cast<T>(sqrt(2 / static_cast<T>(fanIn)));

        std::normal_distribution<T> normal(0, stddev);
        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
    static void HeUniform(const Shape& shape, T* data)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.Row();
        const auto range = static_cast<T>(sqrt(6 / static_cast<T>(fanIn)));

        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

        const auto rowMean = shape.Row() / static_cast<T>(2);
        const auto colMean = shape.Col() / static_cast<T>(2);

        if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.Row(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.Row())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void Zeros(const Shape& shape, T* data)
    {
        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
    static void Ones(const Shape& shape, T* data)
    {
        const auto matrixSize = shape.PaddedMatrixSize();
        const auto batchSize = shape.BatchSize();
        const auto colSize =
            (shape.PadSize() > 0) ? shape.PadSize() : shape.Col();

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
};
} // namespace CubbyDNN
#endif
