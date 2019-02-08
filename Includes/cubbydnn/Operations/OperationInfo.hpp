// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATION_INFO_HPP
#define CUBBYDNN_OPERATION_INFO_HPP

#include <string>

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
    OperationInfo(long id_, size_t inputSize_, size_t outputSize_,
                  const std::string& name_)
        : id(id_), inputSize(inputSize_), outputSize(outputSize_), name(name_)
    {
        // Do nothing
    }

    bool operator==(const OperationInfo& info) const
    {
        return id == info.id && inputSize == info.inputSize &&
               outputSize == info.outputSize && name == info.name;
    }

    bool operator!=(const OperationInfo& info) const
    {
        return !(*this == info);
    }

    long id;
    std::string name;
    size_t inputSize;
    size_t outputSize;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_INFO_HPP