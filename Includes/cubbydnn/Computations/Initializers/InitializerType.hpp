// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_INITIALIZER_HPP
#define CUBBYDNN_INITIALIZER_HPP

#include <cubbydnn/Computations/Initializers/InitializerOp.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
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
    virtual void Initialize(Tensor& tensor, NumberSystem numericType) const = 0;
};

class XavierNormal : public Initializer
{
public:
    XavierNormal() = default;

    void Initialize(Tensor& tensor, NumberSystem numericType) const override
    {
        if (numericType == NumberSystem::Float)
            InitializerOperations::XavierNormal(tensor.TensorShape,
                                                static_cast<float*>(tensor.
                                                    DataPtr),
                                                tensor.PadSize);
        else
            InitializerOperations::XavierNormal(
                tensor.TensorShape, static_cast<int*>(tensor.DataPtr),
                tensor.PadSize);
    }
};

class HeNormal : public Initializer
{
public:
    HeNormal() = default;

    void Initialize(Tensor& tensor, NumberSystem numericType) const override
    {
        if (numericType == NumberSystem::Float)
            InitializerOperations::HeNormal(
                tensor.TensorShape, static_cast<float*>(tensor.DataPtr),
                tensor.PadSize);
        else
            InitializerOperations::HeNormal(
                tensor.TensorShape, static_cast<int*>(tensor.DataPtr),
                tensor.PadSize);
    }
};

class LecunNormal : public Initializer
{
public:
    LecunNormal() = default;

    void Initialize(Tensor& tensor, NumberSystem numericType) const override
    {
        if (numericType == NumberSystem::Float)
            InitializerOperations::LecunNormal(tensor.TensorShape,
                                               static_cast<float*>(tensor.
                                                   DataPtr), tensor.PadSize);
        else
            InitializerOperations::LecunNormal(tensor.TensorShape,
                                               static_cast<int*>(tensor.DataPtr
                                               ), tensor.PadSize);
    }
};

class RandomUniform : public Initializer
{
public:
    RandomUniform(float min, float max);

    void Initialize(Tensor& tensor, NumberSystem numericType) const override
    {
        if (numericType == NumberSystem::Float)
            InitializerOperations::RandomUniform(
                tensor.TensorShape, m_min,
                m_max, static_cast<float*>(tensor.DataPtr),
                tensor.PadSize);
        else
            InitializerOperations::RandomUniform(
                tensor.TensorShape, m_integerMin, m_integerMax,
                static_cast<int*>(tensor.DataPtr), tensor.PadSize);
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

    void Initialize(Tensor& tensor, NumberSystem numericType) const override
    {
        if (numericType == NumberSystem::Float)
            InitializerOperations::RandomNormal(
                tensor.TensorShape, m_min, m_max,
                static_cast<float*>(tensor.DataPtr), tensor.PadSize);
        else
            InitializerOperations::RandomNormal(
                tensor.TensorShape, m_integerMin, m_integerMax,
                static_cast<int*>(tensor.DataPtr), tensor.PadSize);
    }

private:
    int m_integerMin;
    int m_integerMax;
    float m_min;
    float m_max;
};
}

#endif
