// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_INITIALIZEROP_HPP
#define TAKION_INITIALIZEROP_HPP

#include <cmath>
#include <Takion/Utils/Shape.hpp>
#include <Takion/Utils/Span.hpp>
#include <random>

namespace Takion::Compute
{
class InitializerOperations
{
public:
    template <typename T>
    static void RandomNormal(T mean, T stddev, Utils::Span<T> data,
                             std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::normal_distribution<T> normal(mean, stddev);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = normal(engine);
        }
    }

    template <typename T>
    static void RandomUniform(const Shape& shape, T min, T max,
                              Utils::Span<T> data, std::size_t elementSize,
                              std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<T> uniform(min, max);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = uniform(engine);
        }
    }

    template <typename T>
    static void LecunNormal(std::size_t fanIn, Utils::Span<T> data,
                            std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto stddev = static_cast<T>(1 / sqrt(static_cast<T>(fanIn)));
        std::normal_distribution<T> normal(0, stddev);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = normal(engine);
        }
    }

    template <typename T>
    static void LecunUniform(std::size_t fanIn, Utils::Span<T> data,
                             std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto range = static_cast<T>(sqrt(3 / static_cast<T>(fanIn)));
        std::uniform_int_distribution<T> uniform(-range, range);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = uniform(engine);
        }
    }

    template <typename T>
    static void XavierNormal(std::size_t fanIn, std::size_t fanOut,
                             Utils::Span<T> data,
                             std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto stddev =
            static_cast<T>(sqrt(2 / static_cast<T>(fanIn + fanOut)));
        std::normal_distribution<T> normal(0, stddev);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = normal(engine);
        }
    }

    template <typename T>
    static void XavierUniform(std::size_t fanIn, std::size_t fanOut,
                              Utils::Span<T> data, std::size_t elementSize,
                              std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto range =
            static_cast<T>(sqrt(6 / static_cast<T>(fanIn + fanOut)));
        std::uniform_int_distribution<T> uniform(-range, range);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = uniform(engine);
        }
    }

    template <typename T>
    static void HeNormal(std::size_t fanIn, Utils::Span<T> data,
                         std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto stddev = static_cast<T>(sqrt(2 / static_cast<T>(fanIn)));
        std::normal_distribution<T> normal(0, stddev);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = normal(engine);
        }
    }

    template <typename T>
    static void HeUniform(std::size_t fanIn, Utils::Span<T> data,
                          std::size_t elementSize, std::size_t batchSize)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        const auto range = static_cast<T>(sqrt(6 / static_cast<T>(fanIn)));
        std::uniform_int_distribution<T> uniform(-range, range);

#pragma omp parallel for schedule(static)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto elementIdx = batchIdx * elementSize;
            for (std::size_t idx = 0; idx < elementSize; ++idx)
                data[elementIdx + idx] = uniform(engine);
        }
    }
};
} // namespace Takion
#endif