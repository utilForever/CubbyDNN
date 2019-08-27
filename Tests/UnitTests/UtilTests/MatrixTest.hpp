//
// Created by jwkim98 on 8/25/19.
//

#ifndef CUBBYDNN_MATRIXTEST_HPP
#define CUBBYDNN_MATRIXTEST_HPP

#include <cassert>
#include <chrono>
#include <cubbydnn/Computations/Functions/ComputeTensor.hpp>
#include <iostream>
#include "gtest/gtest.h"

namespace UtilTest
{
template <typename T>
static T* CreateMatrix(size_t rowSize, size_t colSize)
{
    //T* ptr = static_cast<T*>(malloc(sizeof(T)*rowSize * colSize));
    T* ptr = new T[sizeof(T)*rowSize * colSize];
    for (size_t count = 0; count < rowSize * colSize; ++count)
    {
        *(ptr + count) = count;
    }
    return ptr;
}
static void MatrixTransposeTest();

}  // namespace CubbyDNN

#endif  // CUBBYDNN_MATRIXTEST_HPP
