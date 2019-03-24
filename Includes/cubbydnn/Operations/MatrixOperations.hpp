// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MATRIX_OPERATIONS_HPP
#define CUBBYDNN_MATRIX_OPERATIONS_HPP

#include <cubbydnn/Operations/Decl/Operation.hpp>

#include <string>

namespace CubbyDNN
{
class MatMulOp : public Operation
{
 public:
    explicit MatMulOp(long id, const std::string& name)
    {
        m_name = name;
        m_type = OpType::MAT_MUL;
        m_id = id;
    }
};

class MatAddOp : public Operation
{
 public:
    explicit MatAddOp(long id, const std::string& name)
    {
        m_name = name;
        m_type = OpType::MAT_ADD;
        m_id = id;
    }
};

class MatDotOp : public Operation
{
 public:
    explicit MatDotOp(long id, const std::string& name, float multiplier)
    {
        m_name = name;
        m_type = OpType::MAT_DOT;
        m_id = id;
        m_multiplier = multiplier;
    }

 private:
    float m_multiplier;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_BASIC_OPERATIONS_HPP