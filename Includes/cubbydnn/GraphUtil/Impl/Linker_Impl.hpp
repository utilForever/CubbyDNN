//
// Created by jwkim98 on 3/24/19.
//

#ifndef CUBBYDNN_LINKER_IMPL_HPP
#define CUBBYDNN_LINKER_IMPL_HPP

#include <cubbydnn/GraphControl/Decl/Linker.hpp>
#include <cubbydnn/Tensors/Decl/TensorObject.hpp>

namespace CubbyDNN {
    template<typename T>
    static std::unique_ptr<TensorObject<T>> PassToTensorObject(
            std::unique_ptr<TensorData<T>> DataToSend,
            std::unique_ptr<TensorObject<T>> TensorToReceive)
    {
        TensorToReceive->m_data = std::move(DataToSend);
        return TensorToReceive;
    }

    template<typename T>
    static std::unique_ptr<Operation<T>>
    PassToOperation(
            std::unique_ptr<TensorData<T>>
    DataToSend,
    std::unique_ptr<Operation<T>>
    OperationToReceive,
    size_t Position
    ) {
    OperationToReceive->
    m_LoadedDataContainer = std::move(DataToSend);
    return
    OperationToReceive;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_IMPL_HPP
