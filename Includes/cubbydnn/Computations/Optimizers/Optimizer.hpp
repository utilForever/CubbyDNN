/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPTIMIZER_HPP
#define CUBBYDNN_OPTIMIZER_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace CubbyDNN::Compute
{
class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer& optimizer) = default;
    Optimizer(Optimizer&& optimizer) noexcept = default;
    Optimizer& operator=(const Optimizer& optimizer) = default;
    Optimizer& operator=(Optimizer&& optimizer) noexcept = default;

    virtual void Optimize(Tensor& tensor, Tensor& delta) = 0;
};

class SGD : public Optimizer
{
public:
    SGD(float epsilon)
        : m_epsilon(-epsilon)
    {
    }

    ~SGD() = default;

    void Optimize(Tensor& tensor, Tensor& delta) override
    {
        ScalarMul(delta, m_epsilon);
        Add(tensor, delta);
    }

private:
    float m_epsilon;
};
}

#endif
