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
#include <cubbydnn/Tensors/Decl/TensorObject.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>


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
    Operation() = default;

    ///Only move constructor is allowed
    Operation(Operation&& operation) noexcept;

    Operation<T>& operator=(Operation&& operation) noexcept;

    std::string GetName() const noexcept;

    OperationInfo GetInfo() const noexcept;

    TensorDataPtr<T> RequestDataFrom(int index);

    void SendDataTo(int index, TensorDataPtr<T> tensorDataPtr);

    void AddOutput(TensorObjectPtr<T> tensorObjectPtr);

    void AddInput(TensorSocketPtr<T> tensorSocketPtr);

 protected:
    /// Disable the default operation
    /// Type of this operation
    OpType m_type = OpType::EMPTY;
    /// OperationInfo class that holds information about this Operation
    OperationInfo m_operationInfo;
    /// contains Data to be used in operation
    std::vector<TensorSocket<T>> m_tensorSocketDeck;
    /// contains tensorObjects going out of this operation
    std::vector<TensorObjectPtr<T>> m_tensorObjectDeck;
};
}  // namespace CubbyDNN

#endif  //CUBBYDNN_OPERATION_HPP