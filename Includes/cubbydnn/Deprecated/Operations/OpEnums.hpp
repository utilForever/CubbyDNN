// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OP_ENUMS_HPP
#define CUBBYDNN_OP_ENUMS_HPP

namespace CubbyDNN
{
enum class OpType
{
    EMPTY,
    PLACEHOLDER,
    WEIGHT,
    MAT_MUL,
    MAT_ADD,
    MAT_DOT,
    RESHAPE,
    CONSTANT,
    WRAPPER
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OP_ENUMS_HPP