// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_STRUCTS_HPP
#define CUBBYDNN_STRUCTS_HPP
#include <atomic>

namespace Takion
{
enum class TaskType
{
    ComputeSource,
    ComputeSink,
    ComputeHidden,
    Copy,
    Join,
    None,
};


enum class Activation
{
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
    Elu,
    Selu,
    Softplus,
    softsign,
    HardSigmoid,
    Linear,
};

enum class Padding
{
    Zeros,
    None,
    Same,
};

enum class OptimizerType
{
    Adam,
    Adagrad,
    SGD,
    Momentum,
};

enum class Loss
{
    CrossEntropy,
    MeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredLogarithmicError,
};

enum class Regularization
{
    L1,
    L2,
};
}  // namespace Takion

#endif
