// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef CUBBYDNN_TENSORSOCKET_HPP
#define CUBBYDNN_TENSORSOCKET_HPP

#include <cubbydnn/GraphUtil/Decl/Sync.hpp>
#include <cubbydnn/Tensors/Decl/TensorData.hpp>

#include <condition_variable>
#include <future>
#include <memory>

/**
 *  This class performs role of socket that receives TensorObjects.
 *  This Class will be stored by Operations, and TensorObjects heading to that
 *  TensorSocket will point to corresponding TensorSocket by unique_ptr to
 * TensorSocket
 *
 */
namespace CubbyDNN
{
template <typename T>
class TensorSocket
{
 public:
    TensorSocket(SyncPtr operationSyncPtr, SyncPtr linkSyncPtr);

    /**
     * Reassigning TensorSocket is disabled
     * @param tensorSocket
     * @return
     */
    TensorSocket& operator=(TensorSocket& tensorSocket) = delete;

    /**
     * Returns dataPtr of current TensorSocket and set m_data to nullptr
     * @return : m_data
     */
    TensorDataPtr<T> MoveDataPtr() const noexcept;

    /**
     * Assigns TensorDataPtr to this tensorSocket
     * Only linker should call this since it decrements operation's atomic counter
     * @param tensorDataPtr : TensorDataPtr to assign
     * @return : True if tensorDataPtr was assigned False if tensorPlug was
     * already assigned
     */
    bool SetDataPtrFromLinker(TensorDataPtr<T> tensorDataPtr);

    /**
     * Assigns TensorDataPtr to this tensorSocket
     * Only operation should call this
     * Notifies Sync that operation has been finished
     * @param tensorDataPtr : TensorDataPtr to assign
     * @return : True if tensorDataPtr was assigned False if tensorPlug was
     * already assigned
     */
    bool SetDataPtrFromOperation(TensorDataPtr<T> tensorDataPtr);

 private:


    TensorDataPtr<T> m_dataPtr = nullptr;

    SyncPtr m_operationSyncPtr;

    SyncPtr m_linkSyncPtr;
};

template <typename T>
using TensorSocketPtr = typename std::unique_ptr<TensorSocket<T>>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TensorSocket_HPP
