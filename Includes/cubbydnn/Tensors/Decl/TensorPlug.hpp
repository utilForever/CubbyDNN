// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_HPP
#define CUBBYDNN_TENSOR_OBJECT_HPP

#include <cubbydnn/GraphUtil/Decl/Sync.hpp>
#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Tensors/TensorShape.hpp>

#include <memory>
#include <mutex>

namespace CubbyDNN
{
//!
//! \brief TensorObject class.
//!
//! This class will represent graph at runtime, and stores actual data used in
//! graph execution.
//!
template <typename T>
class TensorPlug
{
 public:

    TensorPlug(SyncPtr operationSyncPtr, SyncPtr linkSyncPtr);

    /**
     * MoveDataPtr
     * Returns dataPtr of current TensorSocket and set m_data to nullptr
     * @return : m_data
     */
    TensorDataPtr<T> MoveDataPtr() const noexcept;

    /**
     * Assigns TensorDataPtr to this tensorPlug
     * Only linker should call this since it decrements operation's atomic counter
     * @param tensorDataPtr : TensorDataPtr to assign
     * @return : True if tensorDataPtr was assigned False if tensorPlug was
     * already assigned
     */
    bool SetDataPtrFromLinker(TensorDataPtr<T> tensorDataPtr);

    /**
     * Assigns TensorDataPtr to this tensorPlug
     * Only operation should call this since it decrements linker's atomic counter
     * @param tensorDataPtr : TensorDataPtr to assign
     * @return : True if tensorDataPtr was assigned False if tensorPlug was
     * already assigned
     */
    bool SetDataPtrFromOperation(TensorDataPtr<T> tensorDataPtr);

 private:
    /// ptr to Data this TensorObject holds
    TensorDataPtr<T> m_dataPtr = nullptr;
    /// ptr to operationSync
    SyncPtr m_operationSyncPtr;
    /// ptr to linkSync
    SyncPtr m_linkSyncPtr;
};

template <typename T>
using TensorPlugPtr = typename std::unique_ptr<TensorPlug<T>>;
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_HPP