// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "TensorTest.hpp"
#include<cubbydnn/Tensors/Tensor.hpp>
#include <doctest.h>
#include <iostream>

namespace CubbyDNN::Test
{
void TensorCopyTest()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "device0");
    const Shape shape({ 10, 3, 5 });
    std::vector<float> vector(shape.Size());

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor tensor1(shape, device);
    const Tensor tensor2(shape, device, vector);

    tensor1 = tensor2;

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        CHECK(static_cast<float>(i) ==
            *(static_cast<float*>(tensor1.DataPtr) + i));
    }
}

void TensorMoveTest()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "device0");
    const Shape shape({ 10, 3, 5 });
    std::vector<float> vector(shape.Size());

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor tensor1(shape, device);
    Tensor tensor2(shape, device, vector);

    tensor1 = std::move(tensor2);

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        CHECK(static_cast<float>(i) ==
            *(static_cast<float*>(tensor1.DataPtr) + i));
    }
}

void TensorForwardTestWithMove()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "device0");
    const Shape shape({ 10, 3, 5 });
    std::vector<float> vector(shape.Size());

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor tensor1(shape, device);
    Tensor tensor2(shape, device, vector);

    Tensor::ForwardTensorData(tensor2, tensor1);

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        CHECK(static_cast<float>(i) ==
            *(static_cast<float*>(tensor1.DataPtr) + i));
    }
}

void TensorForwardTestWithCopy()
{
    Compute::Device device1(0, Compute::DeviceType::Cpu, "device0");
    Compute::Device device2(1, Compute::DeviceType::Blaze, "device1", 256);
    const Shape shape({ 10, 3, 5 });
    std::vector<float> vector(shape.Size());

    for (std::size_t i = 0; i < shape.Size(); ++i)
    {
        vector.at(i) = static_cast<float>(i);
    }

    Tensor tensor1(shape, device2);
    Tensor tensor2(shape, device1, vector);

    Tensor::ForwardTensorData(tensor2, tensor1);

    auto ans = 0;
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 5; ++k)
            {
                CHECK(static_cast<float>(ans++) ==
                    tensor1.At<float>({i,j,k}));
            }
}
}
