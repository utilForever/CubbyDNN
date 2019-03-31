//
// Created by jwkim98 on 3/24/19.
//

#ifndef CUBBYDNN_LINKER_IMPL_HPP
#define CUBBYDNN_LINKER_IMPL_HPP

#include <cubbydnn/GraphUtil/Decl/Linker.hpp>
#include <cubbydnn/Tensors/Decl/TensorObject.hpp>

namespace CubbyDNN
{
template <typename T>
static std::unique_ptr<TensorObject<T>> PassToTensorObject(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorObject<T>> TensorToReceive)
{
    TensorToReceive->m_data = std::move(DataToSend);
    return std::move(TensorToReceive);
}

template <typename T>
static std::unique_ptr<TensorSocket<T>> PassToOperation(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorSocket<T>> SocketToReceive, size_t Position)
{
    SocketToReceive->SocketTensorData = std::move(DataToSend);
    return std::move(SocketToReceive);
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_IMPL_HPP
