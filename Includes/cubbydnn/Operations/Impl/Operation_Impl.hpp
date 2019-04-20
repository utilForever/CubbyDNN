//
// Created by jwkim98 on 3/25/19.
//

#ifndef CUBBYDNN_OPERATION_IMPL_HPP
#define CUBBYDNN_OPERATION_IMPL_HPP

#include <cubbydnn/Operations/Decl/Operation.hpp>

namespace CubbyDNN
{
template <typename T>
std::string Operation<T>::GetName() const noexcept
{
    return m_operationInfo.name;
}

template <typename T>
Operation<T>::Operation(Operation<T>&& operation) noexcept
    : m_type(operation.m_type),
      m_operationInfo(operation.m_operationInfo),
      m_tensorSocketDeck(std::move(operation.m_tensorSocketDeck)),
      m_tensorObjectDeck(std::move(operation.m_tensorObjectDeck))
{
}

template <typename T>
Operation<T>& Operation<T>::operator=(Operation&& operation) noexcept
{
    m_type = operation.m_type;
    m_operationInfo = operation.m_info;
    m_tensorSocketDeck = std::move(operation.m_tensorSocketDeck);
    m_tensorObjectDeck = std::move(operation.m_tensorObjectDeck);
}

template <typename T>
OperationInfo Operation<T>::GetInfo() const noexcept
{
    return m_operationInfo;
}

template <typename T>
TensorDataPtr<T> Operation<T>::RequestDataFrom(int index)
{
    return m_tensorSocketDeck.at(index).Request();
}

template <typename T>
void Operation<T>::SendDataTo(int index, TensorDataPtr<T> tensorDataPtr)
{
    m_tensorObjectDeck.at(index)->SetData(tensorDataPtr);
}

template <typename T>
void Operation<T>::AddOutput(TensorObjectPtr<T> tensorObjectPtr)
{
    m_tensorObjectDeck.emplace_back(tensorObjectPtr);
}

template <typename T>
void Operation<T>::AddInput(TensorSocketPtr<T> tensorSocketPtr)
{
    m_tensorSocketDeck.emplace_back(tensorSocketPtr);
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_IMPL_HPP
