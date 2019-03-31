/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Linker.hpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_LINKER_HPP
#define CUBBYDNN_LINKER_HPP

#include <cubbydnn/Operations/Decl/Operation.hpp>
#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorObject.hpp>

#include <memory>

namespace CubbyDNN {
/**
 * @brief : Passes
 * @tparam T : Template type for TensorData
 * @param DataToSend : ptr to TensorData which needs to be passed
 * @param TensorToReceive : ptr to ObjectToReceive
 * @return : ptr to TensorObject after passing
 */
    template<typename T>
    static std::unique_ptr<TensorObject<T>> PassToTensorObject(
            std::unique_ptr<TensorData<T>> DataToSend,
            std::unique_ptr<TensorObject<T>> TensorToReceive);

/**
 * @tparam T : Template type for TensorData
 * @param DataToSend : ptr to TensorData which needs to be passed
 * @param ObjectToReceive : ptr to ObjectToReceive
 * @return : ptr to Operation after passing
 */
    template<typename T>
    static std::unique_ptr<Operation<T>> PassToOperation(
            std::unique_ptr<TensorData<T>> DataToSend,
            std::unique_ptr<Operation<T>> OperationToReceive, size_t Position);

    template<typename T>


}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_HPP
