/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Linker.hpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_LINKER_HPP
#define CUBBYDNN_LINKER_HPP

#include <cubbydnn/Operations/Decl/TensorSocket.hpp>
#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorObject.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

#include <memory>

namespace CubbyDNN
{
/**
 * @brief : Passes
 * @tparam T : Template type for TensorData
 * @param DataToSend : ptr to TensorData which needs to be passed
 * @param TensorToReceive : ptr to ObjectToReceive
 * @return : ptr to TensorObject after passing
 */
template <typename T>
static std::unique_ptr<TensorObject<T>> PassToTensorObject(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorObject<T>> TensorToReceive);

/**
 * @tparam T : Template type for TensorData
 * @param DataToSend : ptr to TensorData which needs to be passed
 * @param ObjectToReceive : ptr to ObjectToReceive
 * @return : ptr to Operation after passing
 */
template <typename T>
static std::unique_ptr<TensorSocket<T>> PassToOperation(
    std::unique_ptr<TensorData<T>> DataToSend,
    std::unique_ptr<TensorSocket<T>> SocketToReceive, size_t Position);

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_HPP
