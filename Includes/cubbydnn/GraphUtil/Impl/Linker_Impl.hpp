/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Linker.cpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_LINKER_IMPL_HPP
#define CUBBYDNN_LINKER_IMPL_HPP

#include <cubbydnn/GraphUtil/Decl/Linker.hpp>
#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>

namespace CubbyDNN
{
template <typename T>
Linker<T>::Linker(TensorSocketPtr<T> socketPtr, TensorPlugPtr<T> plugPtr,
                  SyncPtr syncPtr)
    : m_tensorPlugPtr(plugPtr), m_tensorSocketPtr(socketPtr), m_syncPtr(syncPtr)
{
}

template <typename T>
void Linker<T>::Swap()
{
    while (!m_forceFinish)
    {
        m_syncPtr->WaitUntilAllFinish();

        TensorDataPtr<T> plugDataPtr = m_tensorPlugPtr->MoveDataPtr();
        TensorDataPtr<T> socketDataPtr = m_tensorSocketPtr->MoveDataPtr();

        TensorDataPtr<T> temp = std::move(plugDataPtr);
        plugDataPtr = std::move(socketDataPtr);
        socketDataPtr = std::move(temp);

        plugDataPtr->moveReady = false;
        socketDataPtr->moveReady = false;
    }
}

template <typename T>
void Linker<T>::Start()
{
    m_thread = std::move(std::thread(Swap));
}

template <typename T>
void Linker<T>::ForceFinish()
{
    m_syncPtr->ForceFinish();
    if(m_thread.joinable())
        m_thread.join();
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_IMPL_HPP
