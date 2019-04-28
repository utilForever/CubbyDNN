//
// Created by jwkim98 on 3/24/19.
//

#ifndef CUBBYDNN_LINKER_IMPL_HPP
#define CUBBYDNN_LINKER_IMPL_HPP

#include <cubbydnn/GraphUtil/Decl/Linker.hpp>
#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>

namespace CubbyDNN
{
template <typename T>
static std::unique_ptr<TensorPlug<T>> PassToTensorObject(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorPlug<T>> TensorToReceive)
{
    TensorToReceive->m_data = std::move(DataToSend);
    return std::move(TensorToReceive);
}

template <typename T>
static std::unique_ptr<TensorSocket<T>> PassToOperation(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorSocket<T>> SocketToReceive, size_t Position)
{
    SocketToReceive->m_socketTensorData = std::move(DataToSend);
    return std::move(SocketToReceive);
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_IMPL_HPP
