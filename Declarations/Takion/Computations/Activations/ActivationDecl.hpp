// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ACTIVATIONFUNC_HPP
#define CUBBYDNN_ACTIVATIONFUNC_HPP
#include <Takion/Tensors/Tensor.hpp>

namespace Takion::Compute
{
template <typename T>
class ActivationFunc
{
public:
    ActivationFunc() = default;

    ActivationFunc(const ActivationFunc& activationFunction) = default;
    ActivationFunc(ActivationFunc&& activationFunction) noexcept
    = default;
    ActivationFunc& operator=(const ActivationFunc& activationFunction)
    = default;
    ActivationFunc& operator=(ActivationFunc&& activationFunction) noexcept
    = default;
    virtual ~ActivationFunc() = default;

    virtual void Apply(Tensor<T>& input, Tensor<T>& output) const = 0;

    virtual void ApplyDerivative(Tensor<T>& input,
                                 Tensor<T>& output) const = 0;
};

template <typename T>
class ReLU : public ActivationFunc<T>
{
public:
    ReLU() = default;

    ReLU(const ReLU<T>& reLU) = default;
    ReLU(ReLU<T>&& reLU) noexcept = default;
    ReLU& operator=(const ReLU<T>& reLU) = default;
    ReLU& operator=(ReLU<T>&& reLU) noexcept = default;
    void Apply(Tensor<T>& input, Tensor<T>& output) const override;

    void ApplyDerivative(Tensor<T>& input,
                         Tensor<T>& output) const override;
    ~ReLU() override = default;

private:
    [[nodiscard]] static T m_apply(T data)
    {
        if (data > static_cast<T>(0))
            return data;
        return 0;
    }

    [[nodiscard]] static T m_applyDerivative(T data)
    {
        if (data > static_cast<T>(0))
            return static_cast<T>(1);
        return static_cast<T>(0);
    }
};

template <typename T>
class SoftMax : public ActivationFunc<T>
{
public:
    SoftMax() = default;
    ~SoftMax() override = default;

    SoftMax(const SoftMax<T>& softMax) = default;
    SoftMax(SoftMax<T>&& softMax) noexcept = default;
    SoftMax<T>& operator=(const SoftMax<T>& softMax) = default;
    SoftMax<T>& operator=(SoftMax<T>&& softMax) noexcept = default;

    void Apply(Tensor<T>& input, Tensor<T>& output) const override;

    void ApplyDerivative(Tensor<T>& input,
                         Tensor<T>& output) const override;
};
}

#endif
