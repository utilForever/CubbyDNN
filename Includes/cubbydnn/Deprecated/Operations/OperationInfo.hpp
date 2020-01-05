// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATION_INFO_HPP
#define CUBBYDNN_OPERATION_INFO_HPP

#include <string>
#include <utility>

namespace CubbyDNN
{
//!
//! \brief Operation class.
//!
//! This class represents information of operation objects. It used for
//! examining graph structure.
//!
struct OperationInfo
{
 public:
    OperationInfo() = default;
    OperationInfo(long id_, std::string name_, size_t inputSize_,
                  size_t outputSize_) noexcept
        : id(id_),
          name(std::move(name_)),
          inputSize(inputSize_),
          outputSize(outputSize_)
    {
        // Do nothing
    }

    bool operator==(const OperationInfo& info) const noexcept
    {
        return id == info.id && inputSize == info.inputSize &&
               outputSize == info.outputSize && name == info.name;
    }

    bool operator!=(const OperationInfo& info) const noexcept
    {
        return !(*this == info);
    }

    long id = 0;
    std::string name = "";
    size_t inputSize = 0;
    size_t outputSize = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_INFO_HPP