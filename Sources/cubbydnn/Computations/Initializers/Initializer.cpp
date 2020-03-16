// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties

#include <cubbydnn/Computations/Initializers/Initializer.hpp>

namespace CubbyDNN
{
void Initializer::RandomNormal(Tensor& tensor, float mean, float stddev)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();

    if (numberSystem == NumberSystem::Float)
        InitializerOperations::RandomNormal<float>(
            tensor.Info.GetShape(), static_cast<float>(mean),
            static_cast<float>(stddev), static_cast<float*>(tensor.DataPtr));
    else
        throw std::runtime_error("Unsupported Number System");
}

void Initializer::RandomUniform(Tensor& tensor, float min, float max)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::RandomUniform<float>(
            tensor.Info.GetShape(), static_cast<float>(min),
            static_cast<float>(max), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::RandomUniform<int>(
            tensor.Info.GetShape(), static_cast<int>(min),
            static_cast<int>(max), static_cast<int*>(tensor.DataPtr));
}

void Initializer::LecunNormal(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::LecunNormal<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        throw std::runtime_error("Unsupported Number System");
}

void Initializer::XavierNormal(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::XavierNormal<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        throw std::runtime_error("Unsupported Number System");
}

void Initializer::HeNormal(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::HeNormal<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        throw std::runtime_error("Unsupported Number System");
}

void Initializer::LecunUniform(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::LecunUniform<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::LecunUniform<int>(
            tensor.Info.GetShape(), static_cast<int*>(tensor.DataPtr));
}

void Initializer::XavierUniform(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::XavierUniform<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::XavierUniform<int>(
            tensor.Info.GetShape(), static_cast<int*>(tensor.DataPtr));
}

void Initializer::HeUniform(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::HeUniform<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::HeUniform<int>(
            tensor.Info.GetShape(), static_cast<int*>(tensor.DataPtr));
}

void Initializer::Zeros(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::Zeros<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::Zeros<int>(
            tensor.Info.GetShape(), static_cast<int*>(tensor.DataPtr));
}

void Initializer::Ones(Tensor& tensor)
{
    const auto numberSystem = tensor.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        InitializerOperations::Ones<float>(
            tensor.Info.GetShape(), static_cast<float*>(tensor.DataPtr));
    else
        InitializerOperations::Ones<int>(
            tensor.Info.GetShape(), static_cast<int*>(tensor.DataPtr));
}
} // namespace CubbyDNN
