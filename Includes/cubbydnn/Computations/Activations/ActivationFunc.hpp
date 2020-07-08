// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ACTIVATIONFUNC_HPP
#define CUBBYDNN_ACTIVATIONFUNC_HPP
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN::Compute
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

    virtual void Apply(Tensor& input, Tensor& output) const = 0;

    virtual void ApplyDerivative(Tensor& input,
                                 Tensor& output) const = 0;

protected:
    static void m_checkArguments(std::vector<const Tensor&> arguments)
    {
        const auto shape = arguments.at(0).TensorShape;
        const auto numericType = arguments.at(0).NumericType;
        const auto device = arguments.at(0).Device;
        for (const auto& tensor : arguments)
        {
            if (tensor.TensorShape != shape)
                throw std::invalid_argument(
                    "Activation - Tensor shape mismatch");

            if (tensor.NumericType != numericType)
                throw std::invalid_argument(
                    "Activation  - Numeric type mismatch");

            if (tensor.Device != device)
                throw std::invalid_argument("Activation - Device mismatch");
        }
    }
};

template <typename T>
class ReLU : public ActivationFunc<T>
{
public:
    ReLU() = default;

    ReLU(const ReLU& reLU) = default;
    ReLU(ReLU&& reLU) noexcept = default;
    ReLU& operator=(const ReLU& reLU) = default;
    ReLU& operator=(ReLU&& reLU) noexcept = default;
    void Apply(Tensor& input, Tensor& output) const override;

    void ApplyDerivative(Tensor& input,
                         Tensor& output) const override;
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

    SoftMax(const SoftMax& softMax) = default;
    SoftMax(SoftMax&& softMax) noexcept = default;
    SoftMax& operator=(const SoftMax& softMax) = default;
    SoftMax& operator=(SoftMax&& softMax) noexcept = default;

    void Apply(Tensor& input, Tensor& output) const override;

    void ApplyDerivative(Tensor& input,
                         Tensor& output) const override;
};
}

#endif
