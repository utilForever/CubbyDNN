// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_BASIC_OPERATIONS_HPP
#define CUBBYDNN_BASIC_OPERATIONS_HPP

#include <cubbydnn/Deprecated/Operations/Decl/Operation.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>

namespace CubbyDNN
{
class EmptyOp : public Operation
{
 public:
    explicit EmptyOp()
    {
        m_name = "Empty operation";
        m_type = OpType::EMPTY;
    }
};

class ReshapeOp : public Operation
{
 public:
    explicit ReshapeOp(long id, const TensorInfo& shape,
                       const std::string& name)
    {
        m_name = name;
        m_type = OpType::RESHAPE;
        m_id = id;
        m_shape = shape;
    }

 private:
    TensorInfo m_shape;
};

class PlaceholderOp : public Operation
{
 public:
    explicit PlaceholderOp(long id, const TensorInfo& shape,
                           const std::string& name)
    {
        m_name = name;
        m_type = OpType::PLACEHOLDER;
        m_id = id;
        m_shape = shape;
    }

 private:
    TensorInfo m_shape;
};

class WeightOp : public Operation
{
 public:
    explicit WeightOp(long id, const TensorInfo& shape,
                      const std::string& name)
    {
        m_name = name;
        m_type = OpType::WEIGHT;
        m_id = id;
        m_shape = shape;
    }

 private:
    TensorInfo m_shape;
};

class ConstantOp : public Operation
{
 public:
    explicit ConstantOp(long id, const TensorInfo& shape,
                        const std::string& name)
    {
        m_name = name;
        m_type = OpType::CONSTANT;
        m_id = id;
        m_shape = shape;
    }

 private:
    TensorInfo m_shape;
};

class WrapperOp : public Operation
{
 public:
    explicit WrapperOp(long id, const std::string& name)
    {
        m_name = name;
        m_type = OpType::WRAPPER;
        m_id = id;
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_BASIC_OPERATIONS_HPP