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
    static void RandomNormal(const Shape& shape, T mean, T stddev, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::normal_distribution<T> normal(mean, stddev);

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if  (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = normal(engine);
                }
    }

    template <typename T>
    static void RandomUniform(const Shape& shape, T min, T max, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        if constexpr (std::is_integral<T>::value)
        {
            std::uniform_int_distribution<T> uniform(min, max);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(min, max);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void LecunNormal(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto fanIn = shape.NumRows();
        const auto stddev = static_cast<T>(1 / sqrt(static_cast<T>(fanIn)));
        std::normal_distribution<T> normal(0, stddev);

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if(padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = normal(engine);
                }
    }

    template <typename T>
    static void LecunUniform(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto fanIn = shape.NumRows();
        const auto range = static_cast<T>(sqrt(3 / static_cast<T>(fanIn)));

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        if constexpr (std::is_integral<T>::value)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void XavierNormal(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.NumRows();
        const auto fanOut = shape.NumCols();
        const auto stddev = static_cast<T>(sqrt(
            2 / static_cast<T>(fanIn + fanOut)));

        std::normal_distribution<T> normal(0, stddev);

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = normal(engine);
                }
    }

    template <typename T>
    static void XavierUniform(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.NumRows();
        const auto fanOut = shape.NumCols();
        const auto range =
            static_cast<T>(sqrt(6 / static_cast<T>(fanIn + fanOut)));

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        if constexpr (std::is_integral<T>::value)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void HeNormal(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.NumRows();
        const auto stddev = static_cast<T>(sqrt(2 / static_cast<T>(fanIn)));

        std::normal_distribution<T> normal(0, stddev);
        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if(padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = normal(engine);
                }
    }

    template <typename T>
    static void HeUniform(const Shape& shape, T* data, std::size_t padSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());

        const auto fanIn = shape.NumRows();
        const auto range = static_cast<T>(sqrt(6 / static_cast<T>(fanIn)));

        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        if constexpr (std::is_integral<T>::value)
        {
            std::uniform_int_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
        else
        {
            std::uniform_real_distribution<T> uniform(-range, range);
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                    {
                        if (colIdx < shape.NumRows())
                            *(data + batchIdx * matrixSize + rowIdx * colSize +
                              colIdx) = uniform(engine);
                    }
        }
    }

    template <typename T>
    static void Zeros(const Shape& shape, T* data, std::size_t padSize)
    {
        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = static_cast<T>(0);
                }
    }

    template <typename T>
    static void Ones(const Shape& shape, T* data, std::size_t padSize)
    {
        std::size_t matrixSize;
        std::size_t batchSize;
        std::size_t colSize;

        if (padSize)
        {
            matrixSize = shape.NumRows() * padSize;
            batchSize = shape.BatchSize();
            colSize = padSize;
        }
        else
        {
            matrixSize = shape.NumRows() * shape.NumCols();
            batchSize = shape.BatchSize();
            colSize = shape.NumCols();
        }

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t rowIdx = 0; rowIdx < shape.NumRows(); ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < colSize; ++colIdx)
                {
                    if (colIdx < shape.NumRows())
                        *(data + batchIdx * matrixSize + rowIdx * colSize +
                          colIdx) = static_cast<T>(1);
                }
    }
};
} // namespace CubbyDNN
#endif
