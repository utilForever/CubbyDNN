/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Sync.cpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#include <cubbydnn/Utils/Sync.hpp>

namespace CubbyDNN
{
LockFreeSync::LockFreeSync(int maxConnections)
    : m_count(0), m_isOccupied(false), m_maxConnections(maxConnections)
{
}

void LockFreeSync::NotifyFinish()
{
    if (m_count > 0)
        m_count--;
}

bool LockFreeSync::IsReady()
{
    return (m_isOccupied == false);
}

void LockFreeSync::waitUntilReady()
{
    while (m_count > 0)
        ;

    m_isOccupied.exchange(true, std::memory_order_acquire);
    m_count.exchange(m_maxConnections);
}
}  // namespace CubbyDNN