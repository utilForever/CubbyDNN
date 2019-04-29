/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Linker.hpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_LINKER_HPP
#define CUBBYDNN_LINKER_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

#include <memory>

namespace CubbyDNN
{
template <typename T>
class Linker
{
 public:
    Linker(const TensorSocketPtr<T> socketPtr, const TensorPlugPtr<T> plugPtr);
    bool Link() const;

 private:
    TensorPlugPtr<T> m_tensorPlugPtr;
    TensorSocketPtr<T> m_tensorSocketPtr;
    /// Future from tensorPlug
    const std::future<TensorDataPtr<T>> m_PlugFuture;
    /// Future from tensorSocket
    const std::future<TensorDataPtr<T>> m_SocketFuture;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_HPP
