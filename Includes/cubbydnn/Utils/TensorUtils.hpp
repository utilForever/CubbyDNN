// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_UTILS_HPP
#define CUBBYDNN_TENSOR_UTILS_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>

#include <iostream>
#include <string>

namespace CubbyDNN
{
inline bool CheckShape(const TensorInfo& shape,
                       const std::string& opName = "constructor")
{
    // TODO: find way to check if argument was verifiable
    static bool valid = true;

    if (!valid)
    {
        return false;
    }

    if (shape.IsEmpty())
    {
        valid = false;
        std::cout << "Argument GetShape is empty\n";
    }

    if (!valid)
    {
        std::cout << "This Error occurs from operation: " << opName << '\n';
    }

    return valid;
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_UTILS_HPP