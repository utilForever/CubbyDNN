// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_STRUCTS_HPP
#define CUBBYDNN_STRUCTS_HPP
#include <atomic>
#include <cstdlib>

namespace CubbyDNN
{
enum class NumberSystem
{
    Float,
    Int,
};
enum class UnitType
{
    Source,
    Hidden,
    Sink,
    Copy,
};

//! UnitState
//! Wrapper class containing the state and StateNum
//! This represents the execution state of computable Unit
struct UnitState
{
    explicit UnitState();
    /// State number of current
    std::atomic<std::size_t> StateNum = 0;
};

enum class TaskType
{
    ComputeSource,
    ComputeSink,
    ComputeHidden,
    Copy,
    Join,
    None,
};

struct UnitId
{
    UnitType Type;
    std::size_t ID;

    friend bool operator==(const UnitId& lhs, const UnitId& rhs)
    {
        return lhs.Type == rhs.Type && lhs.ID == rhs.ID;
    }

    friend bool operator!=(const UnitId& lhs, const UnitId& rhs)
    {
        return !(lhs == rhs);
    }
};

enum class ActivationType
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

enum class InitializerType
{
    None,
    Zeros,
    Ones,
    RandomNormal,
    RandomUniform,
    TruncatedNormal,
    VarianceScaling,
    Orthogonal,
    Identity,
    LecunUniform,
    Xavier,
    HeNormal,
    LeCunNormal,
    HeUniform,
};

enum class Padding
{
    Zeros,
    None,
    Same,
};

enum class Optimizer
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
}  // namespace CubbyDNN

#endif
