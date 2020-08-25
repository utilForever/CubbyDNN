// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TEST_COMPUTETEST_HPP
#define TAKION_TEST_COMPUTETEST_HPP

#include <Takion/Computations/GEMM/MathKernel.hpp>
//TODO : Move Device.hpp to Util folder
#include <Takion/Computations/Device.hpp>
#include <Takion/Computations/Initializers/InitializerType.hpp>
#include "SolidComputations.hpp"
#include <doctest.h>
#include <type_traits>
#include <iostream>

namespace Takion::Test
{
template <typename T>
void TestMultiply(Compute::Device device)
{
    const auto batchSize = 2;
    const auto numRow = 19;
    const auto numCol = 11;
    const auto numMiddle = 75;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numMiddle });
    Shape shapeB({ numMiddle, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }
    //
    // for (std::size_t i = 0; i < B.BatchSize * B.ElementSize(); ++i)
    // {
    //     std::cout << B.Data[i] << std::endl;
    // }
    //
    // std::cout << "ByteSize : " << sizeof(T) << std::endl;

    const auto t1 = std::chrono::system_clock::now();
    Compute::Multiply(A, B, result);

    const auto t2 = std::chrono::system_clock::now();
    Test::Multiply(A, B, truth);

    const auto t3 = std::chrono::system_clock::now();

    const auto size = result.BatchSize * result.TensorShape.Size();

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        const auto func = result.At(idx);
        const auto ans = truth.At(idx);
        CHECK(func == ans);
    }

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastMultiply1(Compute::Device device)
{
    const auto batchSize = 2;
    const auto numRow = 169;
    const auto numCol = 181;
    const auto numMiddle = 75;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numMiddle });
    Shape shapeB({ numMiddle, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, device);
    Tensor<T> B(shapeB, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Multiply(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Multiply(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    const auto size = result.BatchSize * result.TensorShape.Size();

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        const auto func = result.At(idx);
        const auto ans = truth.At(idx);
        CHECK(func == ans);
    }

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastMultiply2(Compute::Device device)
{
    const auto batchSize = 2;
    const auto numRow = 169;
    const auto numCol = 181;
    const auto numMiddle = 75;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numMiddle });
    Shape shapeB({ numMiddle, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, device);
    Tensor<T> result(shapeOut, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Multiply(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Multiply(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    const auto size = result.BatchSize * result.TensorShape.Size();

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        const auto func = result.At(idx);
        const auto ans = truth.At(idx);
        CHECK(func == ans);
    }

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestTranspose(Compute::Device device)
{
    const auto batchSize = 2;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeIn({ 10, numRow, numCol });
    Shape shapeOut({ 10, numCol, numRow });

    Tensor<T> in(shapeIn, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(in);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(in);
    }

    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    Compute::Transpose(in, result);
    Test::Transpose(in, truth);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t idx = 0; idx < 10; ++idx)
            for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
                for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
                {
                    CHECK(result.At(batchIdx, { idx, colIdx, rowIdx }) ==
                        truth.At(batchIdx, { idx, colIdx, rowIdx }));
                }
}

template <typename T>
void TestShrink(Compute::Device device)
{
    const auto batchSize = 100;
    const auto numRow = 120;
    const auto numCol = 130;
    Compute::Zeros<T> zeroInitializer;

    Shape shapeIn({ 10, numRow, numCol });
    Shape shapeOut({ 10, numRow, numCol });

    Tensor<T> in(shapeIn, batchSize, device);
    Tensor<T> truth(shapeOut, device);
    Tensor<T> result(shapeOut, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(in);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(in);
    }

    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Shrink(in, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Shrink(in, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < 10; ++idx)
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                const auto resultVal = result.At(0, { idx, rowIdx, colIdx });
                const auto truthVal = truth.At(0, { idx, rowIdx, colIdx });
                CHECK(resultVal == truthVal);
            }

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestAdd(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Add(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Add(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
    {
        const auto outputResult = result.At(idx);
        const auto outputTruth = truth.At(idx);
        CHECK(outputResult == outputTruth);
    }

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastAdd1(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Add(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Add(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastAdd2(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Add(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Add(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestSub(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Sub(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Sub(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastSub1(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Sub(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Sub(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastSub2(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Sub(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Sub(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestDot(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Dot(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Dot(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastDot1(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Dot(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Dot(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}

template <typename T>
void TestBroadcastDot2(Compute::Device device)
{
    const auto batchSize = 3;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    if constexpr (std::is_floating_point<T>::value)
    {
        Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                         static_cast<T>(10));
        randomNormalInitializer.Initialize(A);
        randomNormalInitializer.Initialize(B);
    }
    else
    {
        Compute::Ones<T> onesInitializer;
        onesInitializer.Initialize(A);
        onesInitializer.Initialize(B);
    }

    zeroInitializer.Initialize(truth);
    zeroInitializer.Initialize(result);

    const auto t1 = std::chrono::system_clock::now();
    Compute::Dot(A, B, result);
    const auto t2 = std::chrono::system_clock::now();
    Test::Dot(A, B, truth);
    const auto t3 = std::chrono::system_clock::now();

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));

    const auto optimizedMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto normalMulElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Normal version (microseconds) : " << normalMulElapsedTime
        << " Optimized version (microseconds) : "
        << optimizedMulElapsedTime << std::endl;
}
}

#endif
