// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_INITIALIZER_HPP
#define CUBBYDNN_INITIALIZER_HPP

#include <Takion/Computations/Initializers/InitializerOp.hpp>
#include <Takion/Tensors/Tensor.hpp>

namespace Takion
{
class Initializer
{
public:
    Initializer() = default;
    virtual ~Initializer() = default;

    Initializer(const Initializer& other) = default;
    Initializer(Initializer&& other) noexcept = default;
    Initializer& operator=(const Initializer& other) = default;
    Initializer& operator=(Initializer&& other) noexcept = default;
    virtual void Initialize(Tensor& tensor) const = 0;
};

class Zeros : public Initializer
{
public:
    Zeros() = default;

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::Zeros(
                tensor.TensorShape, static_cast<float*>(tensor.DataPtr),
                tensor.Device.PadSize());
        else
            InitializerOperations::Zeros(
                tensor.TensorShape, static_cast<int*>(tensor.DataPtr),
                tensor.Device.PadSize());
    }
};


class XavierNormal : public Initializer
{
public:
    XavierNormal() = default;

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::XavierNormal(
                tensor.TensorShape, static_cast<float*>(tensor.DataPtr),
                tensor.Device.PadSize());
        else
            throw std::invalid_argument(
                "No integer type is available for XavierNormal");
    }
};

class HeNormal : public Initializer
{
public:
    HeNormal() = default;

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::HeNormal(
                tensor.TensorShape, static_cast<float*>(tensor.DataPtr),
                tensor.Device.PadSize());
        else
            throw std::invalid_argument(
                "No integer type is available for HeNormal");
    }
};

class LecunNormal : public Initializer
{
public:
    LecunNormal() = default;

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::LecunNormal(tensor.TensorShape,
                                               static_cast<float*>(tensor.
                                                   DataPtr),
                                               tensor.Device.PadSize());
        else
            throw std::invalid_argument(
                "No integer type is available for LecunNormal");
    }
};

class RandomUniform : public Initializer
{
public:
    RandomUniform(float min, float max);

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::RandomUniform(
                tensor.TensorShape, m_min,
                m_max, static_cast<float*>(tensor.DataPtr),
                tensor.Device.PadSize());
        else
            InitializerOperations::RandomUniform(
                tensor.TensorShape, m_integerMin, m_integerMax,
                static_cast<int*>(tensor.DataPtr), tensor.Device.PadSize());
    }

private:
    int m_integerMin;
    int m_integerMax;
    float m_min;
    float m_max;
};

class RandomNormal : public Initializer
{
public:
    RandomNormal(float min, float max);

    void Initialize(Tensor& tensor) const override
    {
        if (tensor.NumericType == NumberSystem::Float)
            InitializerOperations::RandomNormal(
                tensor.TensorShape, m_min, m_max,
                static_cast<float*>(tensor.DataPtr), tensor.Device.PadSize());
        else
            throw std::invalid_argument(
                "No integer type is available for RandomNormal");
    }

private:
    int m_integerMin;
    int m_integerMax;
    float m_min;
    float m_max;
};
}

#endif
