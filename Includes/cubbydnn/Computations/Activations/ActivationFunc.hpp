// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ACTIVATIONFUNC_HPP
#define CUBBYDNN_ACTIVATIONFUNC_HPP

namespace CubbyDNN::Compute
{
class ActivationFunc
{
public:
    ActivationFunc() = default;
    virtual ~ActivationFunc() = default;


    ActivationFunc(const ActivationFunc& activationFunction) = default;
    ActivationFunc(ActivationFunc&& activationFunction) noexcept
    = default;
    ActivationFunc& operator=(const ActivationFunc& activationFunction)
    = default;
    ActivationFunc& operator=(
        ActivationFunc&& activationFunction) noexcept = default
    ;

    virtual void Forward() = 0;

    virtual void Derivative() = 0;
};
}

#endif
