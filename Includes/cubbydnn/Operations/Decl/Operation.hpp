/**
 *  Copyright (c) 2019 Chris Ohk, Justin Kim
 *  We are making my contributions/submissions to this project solely in our
 *  personal capacity and are not conveying any rights to any intellectual
 *  property of any third parties.
 */

#ifndef CUBBYDNN_OPERATION_HPP
#define CUBBYDNN_OPERATION_HPP

#include <cubbydnn/Operations/OpEnums.hpp>
#include <cubbydnn/Operations/OperationInfo.hpp>
#include <cubbydnn/Tensors/Decl/TensorData.hpp>


#include <string>
#include <vector>
#include <memory>

namespace CubbyDNN
{
//!
//! \brief Operation class.
//!
//! This class is contained in graph structure. It contains information about
//! which operation to execute, tensors that data comes from and tensor to
//! output processed data.
//!

template <typename T>
class Operation
{
 public:
    const std::string& GetName() const noexcept;

    long GetID() const noexcept;

    const std::vector<long>& GetInputTensors() const noexcept;

    const std::vector<long>& GetOutputTensors() const noexcept;

    void AddInputTensor(long tensorID);

    void AddOutputTensor(long tensorID);

    std::size_t GetNumOfInputTensors() const noexcept;

    std::size_t GetNumOfOutputTensors() const noexcept;

    unsigned GetProcessCount() const noexcept;

    void IncrementProcessCount() noexcept;

    OperationInfo GetInfo() const;

 protected:
    /// Disable the default operation
    explicit Operation() = default;
    /// Name of this Operation
    std::string m_name;
    /// Type of this operation
    OpType m_type = OpType::EMPTY;
    /// unique id of this operation
    long m_id = 0;
    /// contains Data to be used in operation
    std::vector<std::unique_ptr<TensorData<T>>> m_LoadedDataContainer;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_HPP