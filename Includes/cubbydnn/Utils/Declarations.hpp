// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_STRUCTS_HPP
#define CUBBYDNN_STRUCTS_HPP
#include <atomic>

namespace CubbyDNN
{
//! Type and Shape declrations

enum class NumberSystem
{
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
};

enum class ShapeAlignment
{
    // Batch, Channel, height, Width
    NCHW,
    // Batch, Height Channel, Width
    NHCW,
    // Batch, Row, Column
    NRC,
    // Batch, Column, Row
    NCR,
    // Batch, Row
    NR,
    // Batch, Column
    NC,
    // Not specified
    None,
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

struct Shape
{
    Shape(size_t batch, size_t channel, size_t row, size_t col)
        : Batch(batch),
          Channel(channel),
          Row(row),
          Col(col)
    {
    }

    Shape(size_t channel, size_t row, size_t col)
        : Channel(channel),
          Row(row),
          Col(col)
    {
    }

    Shape(size_t row, size_t col)
        : Row(row),
          Col(col)
    {
    }

    size_t Batch = 0;
    size_t Channel = 0;
    size_t Row = 0;
    size_t Col = 0;

    friend bool operator==(const Shape& lhs, const Shape& rhs)
    {
        return lhs.Batch == rhs.Batch && lhs.Row == rhs.Row &&
               lhs.Channel == rhs.Channel && lhs.Col == rhs.Col;
    }

    friend bool operator!=(const Shape& lhs, const Shape& rhs)
    {
        return !(lhs == rhs);
    }

    size_t GetTotalSize() const
    {
        return Batch * Row * Channel * Col;
    }
};

//! UnitState
//! Wrapper class containing the state and StateNum
//! This represents the execution state of computable Unit
struct UnitState
{
    explicit UnitState();
    /// State number of current
    std::atomic<std::size_t> StateNum = 0;
    /// True if unit is already in the task queue
    std::atomic<bool> IsBusy = false;
};

enum class TaskType
{
    ComputeSource,
    ComputeSink,
    ComputeIntermediate,
    Copy,
    Join,
    None,
};

struct UnitIdentifier
{
    UnitType Type;
    size_t ID;
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

enum class Initializer
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
