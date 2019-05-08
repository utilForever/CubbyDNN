/**
 * @file : TensorSocket.hpp
 * @author : Justin Kim
 *
 */

#ifndef CUBBYDNN_TENSORSOCKET_HPP
#define CUBBYDNN_TENSORSOCKET_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>

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
    TensorSocket();

    /**
     * Copying TensorSocket is disabled
     * @param tensorSocket
     */
    TensorSocket(TensorSocket<T>& tensorSocket) = delete;

    /**
     * Reassigning TensorSocket is disabled
     * @param tensorSocket
     * @return
     */
    TensorSocket& operator=(TensorSocket& tensorSocket) = delete;

    bool ReceiveData();

    TensorDataPtr<T> GetDataPtr() const noexcept;

    bool SetDataPtr(TensorDataPtr<T> tensorDataPtr);

    std::future<TensorData<T>> GetFuture();

 private:
    std::promise<TensorDataPtr<T>> m_promiseSend;

    TensorDataPtr<T> m_tensorDataPtr = nullptr;
};

template <typename T>
using TensorSocketPtr = typename std::unique_ptr<TensorSocket<T>>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TensorSocket_HPP
