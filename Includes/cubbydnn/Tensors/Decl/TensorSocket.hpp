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
 *  TensorSocket will point to corresponding TensorSocket by unique_ptr to TensorSocket
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

    /**
     * Sets TensorData in order to pass it to TensorSocket
     * This Method should be called by TensorObject
     * @param tensorDataPtr :
     */
    void SetData(TensorDataPtr<T> tensorDataPtr);

    /**
     * Waits until data is available and
     * brings Data from connected TenorObject
     * @return : requested TensorDataPtr
     */
    TensorDataPtr<T> Request();

    /**
     * Checks if data is available and
     * brings Data from connected TensorObjects
     * returns nullptr immediately if data is unavailable
     * @return : requested TensorDataPtr if available, nullptr otherwise
     */
    TensorDataPtr<T> TryRequest();

 private:


    TensorDataPtr<T> SocketTensorData;
    std::future<TensorDataPtr<T>> m_futureReceive;
    std::promise<TensorDataPtr<T>> m_promiseSend;
};

template <typename T>
using TensorSocketPtr = typename std::unique_ptr<TensorSocket<T>>;

}  // namespace CubbyDNN

#endif  //CUBBYDNN_TensorSocket_HPP
