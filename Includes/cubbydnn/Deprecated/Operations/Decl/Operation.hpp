/**
 *  Copyright (c) 2019 Chris Ohk, Justin Kim
 *  We are making my contributions/submissions to this project solely in our
 *  personal capacity and are not conveying any rights to any intellectual
 *  property of any third parties.
 */

#ifndef CUBBYDNN_OPERATION_HPP
#define CUBBYDNN_OPERATION_HPP

#include <cubbydnn/Computations/Decl/Interfaces.hpp>
#include <cubbydnn/Utils/Sync.hpp>
#include <cubbydnn/Deprecated/Operations/OpEnums.hpp>
#include <cubbydnn/Deprecated/Operations/OperationInfo.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Utils/DoubleBuffer.hpp>

#include <memory>
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
class Operation : virtual IExecutable
{
 public:
    Operation(SyncPtr operationSyncPtr, std::unique_ptr<ComputationUnit<T>> computationUnitPtr);

    /// Only move constructor is allowed
    Operation(Operation&& operation) noexcept;

    Operation<T>& operator=(Operation&& operation) noexcept;

    std::string GetName() const noexcept;

    OperationInfo GetInfo() const noexcept;

    void AddInput(TensorSocketPtr<T> tensorSocketPtr);

    void Start() final;

    void Finish() final;

 protected:
    /// Disable the default operation
    /// Type of this operation
    OpType m_type = OpType::EMPTY;
    /// OperationInfo class that holds information about this Operation
    OperationInfo m_operationInfo;
    /// contains Data to be used in operation
    std::vector<TensorSocketPtr<T>> m_tensorSocketDeck;
    /// ptr to Sync
    SyncPtr m_operationSyncPtr;
    /// Computation unit for running computation assigned for this operation
    std::unique_ptr<ComputationUnit<T>> m_computationUnitPtr;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_HPP