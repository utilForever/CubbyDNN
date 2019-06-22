//
// Created by jwkim98 on 6/22/19.
//
#include <cubbydnn/Utils/SharedPtr-impl.hpp>

#include "gtest/gtest.h"
#include <deque>
#include <thread>

#ifndef CUBBYDNN_SHAREDPTRTESTS_HPP
#define CUBBYDNN_SHAREDPTRTESTS_HPP

namespace CubbyDNN
{
    void CopyandDestruct(SharedPtr<int>&& ptr);

    void ConcurrentCopy(int spawnNum, int maxRefCount);
}

#endif //CUBBYDNN_SHAREDPTRTESTS_HPP
