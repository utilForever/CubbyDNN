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
    None,
    Source,
    Hidden,
    Sink,
    Copy,
    Constant,
    Variable,
    PlaceHolder,
    Add,
    Mul,
    Dense,
    Activate,
    Dropout,
    Regularize,
    Reshape,
    Conv,
    MaxPool,
    AveragePool,
    Undefined,
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

struct UnitIdentifier
{
    UnitType Type;
    std::size_t ID;
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

enum class Regularization
{
    L1,
    L2,
};
} // namespace CubbyDNN

#endif
