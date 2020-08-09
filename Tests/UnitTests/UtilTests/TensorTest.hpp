// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TEST_TENSORTEST_HPP
#define TAKION_TEST_TENSORTEST_HPP

#include<Takion/Tensors/Tensor.hpp>
#include <doctest.h>
#include <chrono>
#include <iostream>

namespace Takion::Test
{
template <typename T>
void TensorCopy()
{
    Compute::Device device(0, Compute::DeviceType::CPU, "device0");

    const auto numRow = 3;
    const auto numCol = 5;
    const auto batchSize = 10;
    const Shape shape({ numRow, numCol });
    const auto totalSize = shape.Size() * batchSize;
    std::vector<T> vector(totalSize);

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor<T> tensor1(shape, batchSize, device);
    const Tensor<T> tensor2(shape, batchSize, device, vector);

    tensor1 = tensor2;

    for (std::size_t idx = 0; idx < totalSize; ++idx)
    {
        CHECK(static_cast<T>(idx) == tensor1.At(idx));
    }
}

template <typename T>
void TensorCopyBetweenDevice_1()
{
    Compute::Device device0(0, Compute::DeviceType::CPU, "device0");
    Compute::Device device1(1, Compute::DeviceType::GPU, "device1");

    const auto numRow = 3;
    const auto numCol = 5;
    const auto batchSize = 10;
    const Shape shape({ numRow, numCol });
    const auto totalSize = shape.Size() * batchSize;
    std::vector<T> vector(totalSize);

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        vector.at(i) = static_cast<T>(i);
    }

    Tensor<T> tensor1(shape, batchSize, device0);
    const Tensor<T> tensor2(shape, batchSize, device1, vector);

    tensor1 = tensor2;

    for (std::size_t idx = 0; idx < totalSize; ++idx)
    {
        CHECK(static_cast<T>(idx) == tensor1.At(idx));
    }
}

template <typename T>
void TensorCopyTestBetweenDevice_2()
{
    Compute::Device device0(0, Compute::DeviceType::GPU, "device0");
    Compute::Device device1(1, Compute::DeviceType::CPU, "device1");

    const auto numRow = 3;
    const auto numCol = 5;
    const auto batchSize = 10;
    const Shape shape({ numRow, numCol });
    const auto totalSize = shape.Size() * batchSize;
    std::vector<T> vector(totalSize);

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        vector.at(i) = static_cast<T>(i);
    }

    Tensor<T> tensor1(shape, batchSize, device0);
    const Tensor<T> tensor2(shape, batchSize, device1, vector);

    tensor1 = tensor2;

    for (std::size_t idx = 0; idx < totalSize; ++idx)
    {
        CHECK(static_cast<T>(idx) ==
            tensor1.At(idx));
    }
}

template <typename T>
void TensorCopyDataSmall()
{
    Compute::Device device(0, Compute::DeviceType::CPU, "device0");

    const Shape shape({ 3 });
    const auto batchSize = 10;
    const auto totalSize = shape.Size() * batchSize;
    std::vector<float> vector(totalSize);

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor<T> sourceTensor(shape, batchSize, device, vector);
    Tensor<T> destTensor(shape, batchSize, device);

    const auto start = std::chrono::system_clock::now();
    Tensor<T>::CopyTensorData(sourceTensor, destTensor);
    const auto end = std::chrono::system_clock::now();

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();

    std::cout << "Elapsed time (microseconds): " << elapsedTime << std::endl;

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        CHECK(static_cast<T>(i) == destTensor.At(i));
    }
}

template <typename T>
void TensorCopyDataLarge()
{
    Compute::Device device(0, Compute::DeviceType::CPU, "device0");

    const Shape shape({ 300, 500, 400 });
    const auto batchSize = 100;
    const auto totalSize = shape.Size() * batchSize;
    std::vector<float> vector(totalSize);

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor<T> sourceTensor(shape, batchSize, device, vector);
    Tensor<T> destTensor(shape, batchSize, device);

    const auto start = std::chrono::system_clock::now();
    Tensor<T>::CopyTensorData(sourceTensor, destTensor);
    const auto end = std::chrono::system_clock::now();

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();

    std::cout << "Elapsed time (microseconds): " << elapsedTime << std::endl;

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        CHECK(static_cast<T>(i) == destTensor.At(i));
    }
}

template <typename T>
void TensorMoveData()
{
    Compute::Device device(0, Compute::DeviceType::CPU, "device0");

    const Shape shape({ 300, 500, 400 });
    const auto batchSize = 100;
    const auto totalSize = shape.Size() * batchSize;
    std::vector<float> vector(totalSize);

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor<T> sourceTensor(shape, batchSize, device, vector);
    Tensor<T> destTensor(shape, batchSize, device);

    const auto start = std::chrono::system_clock::now();
    Tensor<T>::MoveTensorData(sourceTensor, destTensor);
    const auto end = std::chrono::system_clock::now();

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();

    std::cout << "Elapsed time (microseconds): " << elapsedTime << std::endl;

    for (std::size_t i = 0; i < totalSize; ++i)
    {
        CHECK(static_cast<T>(i) == destTensor.At(i));
    }
}
}


#endif
