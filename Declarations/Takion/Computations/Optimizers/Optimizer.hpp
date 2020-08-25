// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_OPTIMIZER_HPP
#define TAKION_OPTIMIZER_HPP

#include <Takion/Tensors/Tensor.hpp>
#include <Takion/Computations/GEMM/MathKernel.hpp>

namespace Takion::Compute
{
template <typename T>
class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer<T>& optimizer) = default;
    Optimizer(Optimizer<T>&& optimizer) noexcept = default;
    Optimizer<T>& operator=(const Optimizer<T>& optimizer) = default;
    Optimizer<T>& operator=(Optimizer<T>&& optimizer) noexcept = default;

    virtual void Optimize(Tensor<T>& tensor, Tensor<T>& delta) = 0;
};

//! Stochastic gradient descent
template <typename T>
class SGD : public Optimizer<T>
{
public:
    SGD(T epsilon)
        : m_epsilon(epsilon)
    {
    }

    SGD(const SGD<T>& sgd) = default;
    SGD(SGD<T>&& sgd) noexcept = default;
    SGD& operator=(const SGD<T>& sgd) = default;
    SGD& operator=(SGD<T>&& sgd) noexcept = default;
    ~SGD() = default;

    void Optimize(Tensor<T>& tensor, Tensor<T>& update) override
    {
        Compute::ScalarMul(update, m_epsilon);
        Compute::Add(tensor, update, tensor);
    }

private:
    T m_epsilon;
};
}

#endif
