// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "UtilTests/SharedPtrTests.hpp"
#include "UtilTests/WeakPtrTests.hpp"
#include "UtilTests/TensorTest.hpp"
#include "ComputeTests/ComputeTest.hpp"
#include <doctest.h>
#include <iostream>

namespace Takion::Test
{
TEST_CASE("Tensor test")
{
    SUBCASE("Tensor Copy")
    {
        SUBCASE("float")
        {
            TensorCopy<float>();
        }
        SUBCASE("int")
        {
            TensorCopy<int>();
        }
    }

    SUBCASE("Copy between devices")
    {
        SUBCASE("CPU to GPU")
        {
            TensorCopyBetweenDevice_1<float>();
            TensorCopyBetweenDevice_1<int>();
        }

        SUBCASE("GPU to CPU")
        {
            TensorCopyBetweenDevice_2<float>();
            TensorCopyBetweenDevice_2<int>();
        }
    }

    SUBCASE("Tensor Move test")
    {
        TensorMoveData<float>();
        TensorMoveData<int>();
    }
    SUBCASE("Tensor forward by copy")
    {
        SUBCASE("Small")
        {
            TensorCopyDataSmall<float>();
            TensorCopyDataSmall<int>();
        }

        SUBCASE("Large")
        {
            TensorCopyDataLarge<float>();
            TensorCopyDataLarge<int>();
        }
    }

    SUBCASE("Tensor forward by move")
    {
        TensorMoveData<float>();
        TensorMoveData<int>();
    }
}

TEST_CASE("Computation test")
{
    SUBCASE("CPU")
    {
        Compute::Device device(0, Compute::DeviceType::CPU, "device");
        SUBCASE("Multiply")
        {
            SUBCASE("float")
            {
                TestMultiply<float>(device);
            }
            SUBCASE("int")
            {
                TestMultiply<int>(device);
            }
        }

        SUBCASE("Add")
        {
            SUBCASE("float")
            {
                TestAdd<float>(device);
            }
            SUBCASE("int")
            {
                TestAdd<int>(device);
            }
        }

        SUBCASE("Shrink")
        {
            SUBCASE("float")
            {
                TestShrink<float>(device);
            }
            SUBCASE("int")
            {
                TestShrink<int>(device);
            }
        }

        SUBCASE("Transpose")
        {
            SUBCASE("float")
            {
                TestTranspose<float>(device);
            }
            SUBCASE("int")
            {
                TestTranspose<int>(device);
            }
        }
    }
}

TEST_CASE("ConcurrentCopy - small")
{
    //! Spawn 10 threads and copy SharedPtr
    //! Check if reference counter successfully returns 1 at the end
    ConcurrentCopy(10, 100);
}

TEST_CASE("ConcurrentCopy - large")
{
    //! Spawn 100 threads and copy SharedPtr
    //!  Check if reference counter successfully returns 1 at the end
    ConcurrentCopy(100, 100);
}

TEST_CASE("Weakptr - OwnershipTransfer")
{
    SimpleOwnershipTransfer();
}
}
