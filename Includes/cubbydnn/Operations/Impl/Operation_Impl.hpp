//
// Created by jwkim98 on 3/25/19.
//

#ifndef CUBBYDNN_OPERATION_IMPL_HPP
#define CUBBYDNN_OPERATION_IMPL_HPP

#include <cubbydnn/Operations/Decl/Operation.hpp>

namespace CubbyDNN
{
template <typename T>
Operation<T>::Operation(SyncPtr operationSyncPtr,
        std::unique_ptr<ComputationUnit<T>> computationUnitPtr)
    : m_operationSyncPtr(operationSyncPtr), m_computationUnitPtr(std::move(computationUnitPtr))
{
}

template <typename T>
Operation<T>::Operation(Operation<T>&& operation) noexcept
    : m_type(operation.m_type),
      m_operationInfo(operation.m_operationInfo),
      m_tensorSocketDeck(std::move(operation.m_tensorSocketDeck)),
      m_tensorPlugDeck(std::move(operation.m_tensorPlugDeck)),
      m_operationSyncPtr(std::move(operation.m_operationSyncPtr))
{
}

template <typename T>
std::string Operation<T>::GetName() const noexcept
{
    return m_operationInfo.name;
}

template <typename T>
Operation<T>& Operation<T>::operator=(Operation&& operation) noexcept
{
    m_type = operation.m_type;
    m_operationInfo = operation.m_info;
    m_tensorSocketDeck = std::move(operation.m_tensorSocketDeck);
    m_tensorPlugDeck = std::move(operation.m_tensorPlugDeck);
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
void Operation<T>::AddOutput(TensorPlugPtr<T> tensorObjectPtr)
{
    m_tensorPlugDeck.emplace_back(tensorObjectPtr);
}

template <typename T>
void Operation<T>::AddInput(TensorSocketPtr<T> tensorSocketPtr)
{
    m_tensorSocketDeck.emplace_back(tensorSocketPtr);
}

template <typename T>
void Operation<T>::Start()
{
    assert(m_tensorSocketDeck.size() == m_computationUnitPtr.GetInputSize());
    assert(m_tensorPlugDeck.size() == m_computationUnitPtr.GetoutputSize());

    m_operationSyncPtr->WaitUntilAllFinish();

    for (size_t idx = 0; idx < m_tensorSocketDeck.size(); ++idx)
    {
        m_computationUnitPtr->SetInput(m_tensorSocketDeck.at(idx)->MoveDataPtr());
    }

    for (size_t idx = 0; idx < m_tensorPlugDeck.size(); ++idx)
    {
        m_computationUnitPtr->SetOutput(m_tensorPlugDeck.at(idx)->MoveDataPtr());
    }

    m_computationUnitPtr->Compute();

    for (size_t idx = 0; idx < m_tensorSocketDeck.size(); ++idx)
    {
        m_tensorSocketDeck.at(idx)->SetDataPtrFromOperation(m_computationUnitPtr->GetInput());
    }

    for (size_t idx = 0; idx < m_tensorPlugDeck.size(); ++idx)
    {
        m_tensorSocketDeck.at(idx)->SetDataPtrFromOperation(m_computationUnitPtr->GetInput());
    }
}

template <typename T>
void Operation<T>::Finish()
{
    m_operationSyncPtr->ForceFinish();
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_IMPL_HPP
