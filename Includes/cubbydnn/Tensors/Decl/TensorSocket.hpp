//
// Created by jwkim98 on 3/31/19.
//

#ifndef CUBBYDNN_TENSORSOCKET_HPP
#define CUBBYDNN_TENSORSOCKET_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>

#include <future>
#include <memory>

/**
 *  @brief : This class performs role of socket that receives TensorObjects.
 *  This Class will be stored by Operations, and TensorObjects heading to that
 *  Operation will point to corresponding TensorSocket by unique_ptr to
 * TensorSocket
 */
namespace CubbyDNN
{
template <typename T>
class TensorSocket
{
 public:
    TensorSocket();

    void ReceiveData(TensorDataPtr<T> tensorDataPtr);

    TensorDataPtr<T> Request();

    TensorDataPtr<T> TryRequest();

 private:
    TensorDataPtr<T> SocketTensorData;
    std::future<TensorDataPtr<T>> m_futureReceive;
    std::promise<TensorDataPtr<T>> m_promiseSend;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSORSOCKET_HPP
