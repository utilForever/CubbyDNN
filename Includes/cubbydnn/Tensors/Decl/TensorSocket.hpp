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

    /**
     * Sets m_promiseSend to tensorDataPtr so it can be requested from
     * Operations This Method should be called by TensorObject
     * @param tensorDataPtr : ptr to be set
     */
    void SendData(TensorDataPtr<T> tensorDataPtr);

    /**
     * Attempts to set m_promiseSend
     * @param tensorDataPtr
     * @return : true if Send succeeded false otherwise
     */
    bool TrySendData(TensorDataPtr<T> tensorDataPtr);

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
    std::future<TensorDataPtr<T>> m_futureReceive;
    std::promise<TensorDataPtr<T>> m_promiseSend;

    std::mutex m_mtx;
    std::unique_lock<std::mutex> m_lock;
    std::condition_variable m_cond;

    /// Indicates whether m_promiseSend is ready to be set
    std::atomic_bool m_updateReady = false;
};

template <typename T>
using TensorSocketPtr = typename std::unique_ptr<TensorSocket<T>>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TensorSocket_HPP
