// Copyright(c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_INITIALIZERTYPE_HPP
#define TAKION_INITIALIZERTYPE_HPP

#include <Takion/Computations/Initializers/InitializerOp.hpp>
#include <Takion/Computations/GEMM/MathKernel.hpp>
#include <Takion/Tensors/Tensor.hpp>

namespace Takion::Compute
{
template <typename T>
class Initializer
{
public:
    Initializer() = default;
    virtual ~Initializer() = default;

    Initializer(const Initializer<T>& initializer) = default;
    Initializer(Initializer<T>&& initializer) noexcept = default;
    Initializer& operator=(const Initializer<T>& initializer) = default;
    Initializer& operator=(Initializer<T>&& initializer) noexcept = default;
    virtual void Initialize(Tensor<T>& tensor) const = 0;
};

template <typename T>
class Zeros : public Initializer<T>
{
public:
    Zeros() = default;

    void Initialize(Tensor<T>& tensor) const override
    {
        Compute::Set(tensor, static_cast<T>(0));
    }
};

template <typename T>
class Ones : public Initializer<T>
{
 public:
    Ones() = default;

    void Initialize(Tensor<T>& tensor) const override
    {
        Compute::Set(tensor, static_cast<T>(1));
    }
};

template <typename T>
class XavierNormal : public Initializer<T>
{
public:
    XavierNormal() = default;

    void Initialize(Tensor<T>& tensor) const override
    {
        InitializerOperations::XavierNormal(
            tensor.TensorShape, static_cast<float*>(tensor.DataPtr),
            tensor.Device.PadSize());
    }
};

template <typename T>
class HeNormal : public Initializer<T>
{
public:
    HeNormal() = default;

    void Initialize(Tensor<T>& tensor) const override
    {
        InitializerOperations::HeNormal<T>(
            tensor.TensorShape, tensor.DataPtr,
            tensor.Device.PadSize());
    }
};

template <typename T>
class LecunNormal : public Initializer<T>
{
public:
    LecunNormal() = default;

    void Initialize(Tensor<T>& tensor) const override
    {
        InitializerOperations::LecunNormal<T>(tensor.TensorShape,
                                              tensor.DataPtr,
                                              tensor.Device.PadSize());
    }
};

template <typename T>
class RandomUniform : public Initializer<T>
{
public:
    RandomUniform(float min, float max)
        : Initializer<T>(),
          m_min(min),
          m_max(max)
    {
    }

    void Initialize(Tensor<T>& tensor) const override
    {
        InitializerOperations::RandomUniform<T>(
            tensor.TensorShape, m_min,
            m_max, tensor.DataPtr,
            tensor.Device.PadSize());
    }

private:
    float m_min;
    float m_max;
};

template <typename T>
class RandomNormal : public Initializer<T>
{
public:
    RandomNormal(float min, float max)
        : Initializer<T>(),
          m_min(min),
          m_max(max)
    {
    }

    void Initialize(Tensor<T>& tensor) const override
    {
        InitializerOperations::RandomNormal<T>(
            m_min, m_max, tensor.DataPtr,
            , );
    }

private:
    T m_min;
    T m_max;
};
}

#endif
