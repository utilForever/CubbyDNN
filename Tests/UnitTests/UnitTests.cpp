// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "UtilTests/SharedPtrTests.hpp"
#include "UtilTests/WeakPtrTests.hpp"
#include "UtilTests/TensorTest.hpp"
#include "ComputeTests/ComputeTest.hpp"
#include "GraphTest/SimpleGraphTest.hpp"
#include <doctest.h>
#include <iostream>

namespace Takion::Test
{
// TEST_CASE("Tensor test")
// {
//     SUBCASE("Tensor Copy")
//     {
//         SUBCASE("float")
//         {
//             TensorCopy<float>();
//         }
//         SUBCASE("int")
//         {
//             TensorCopy<int>();
//         }
//     }
//
//     SUBCASE("Copy between devices")
//     {
//         SUBCASE("CPU to GPU")
//         {
//             TensorCopyBetweenDevice_1<float>();
//             TensorCopyBetweenDevice_1<int>();
//         }
//
//         SUBCASE("GPU to CPU")
//         {
//             TensorCopyBetweenDevice_2<float>();
//             TensorCopyBetweenDevice_2<int>();
//         }
//     }
//
//     SUBCASE("Tensor Move test")
//     {
//         TensorMoveData<float>();
//         TensorMoveData<int>();
//     }
//     SUBCASE("Tensor forward by copy")
//     {
//         SUBCASE("Small")
//         {
//             TensorCopyDataSmall<float>();
//             TensorCopyDataSmall<int>();
//         }
//
//         SUBCASE("Large")
//         {
//             TensorCopyDataLarge<float>();
//             TensorCopyDataLarge<int>();
//         }
//     }
//
//     SUBCASE("Tensor forward by move")
//     {
//         TensorMoveData<float>();
//         TensorMoveData<int>();
//     }
// }
//
// TEST_CASE("Computation test")
// {
//     SUBCASE("CPU")
//     {
//         Compute::Device device(0, Compute::DeviceType::CPU, "device");
//         SUBCASE("Multiply")
//         {
//             SUBCASE("float")
//             {
//                 std::cout << "TensorMultiply - float" << std::endl;
//                 TestMultiply<float>(device);
//             }
//             SUBCASE("int")
//             {
//                 std::cout << "TensorMultiply - int" << std::endl;
//                 TestMultiply<int>(device);
//             }
//             SUBCASE("BroadcastMultiply - float")
//             {
//                 std::cout << "TensorBroadcastMultiply - float" << std::endl;
//                 TestBroadcastMultiply1<float>(device);
//                 TestBroadcastMultiply2<float>(device);
//             }
//             SUBCASE("BroadcastMultiply - int")
//             {
//                 std::cout << "TensorBroadcastMultiply - int" << std::endl;
//                 TestBroadcastMultiply1<int>(device);
//                 TestBroadcastMultiply2<int>(device);
//             }
//         }
//
//         SUBCASE("Add")
//         {
//             SUBCASE("float")
//             {
//                 std::cout << "TensorAdd - float" << std::endl;
//                 TestAdd<float>(device);
//             }
//             SUBCASE("int")
//             {
//                 std::cout << "TensorAdd - int" << std::endl;
//                 TestAdd<int>(device);
//             }
//             SUBCASE("BroadCast - float")
//             {
//                 std::cout << "TensorBroadcastAdd - float" << std::endl;
//                 TestBroadcastAdd1<float>(device);
//                 TestBroadcastAdd2<float>(device);
//             }
//             SUBCASE("BroadCast - int")
//             {
//                 std::cout << "TensorBroadcastAdd - int" << std::endl;
//                 TestBroadcastAdd1<int>(device);
//                 TestBroadcastAdd2<int>(device);
//             }
//         }
//
//         SUBCASE("Shrink")
//         {
//             SUBCASE("float")
//             {
//                 std::cout << "TensorShrink - float" << std::endl;
//                 TestShrink<float>(device);
//             }
//             SUBCASE("int")
//             {
//                 std::cout << "TensorShrink - int" << std::endl;
//                 TestShrink<int>(device);
//             }
//         }
//
//         SUBCASE("Dot")
//         {
//             SUBCASE("float")
//             {
//                 std::cout << "TensorDot - float" << std::endl;
//                 TestDot<float>(device);
//             }
//             SUBCASE("int")
//             {
//                 std::cout << "TensorDot - int" << std::endl;
//                 TestDot<int>(device);
//             }
//             SUBCASE("BroadCast - float")
//             {
//                 std::cout << "TensorBroadcastDot - float" << std::endl;
//                 TestBroadcastDot1<float>(device);
//                 TestBroadcastDot2<float>(device);
//             }
//             SUBCASE("BroadCast - int")
//             {
//                 std::cout << "TensorBroadcastDot - int" << std::endl;
//                 TestBroadcastDot1<int>(device);
//                 TestBroadcastDot2<int>(device);
//             }
//         }
//
//         SUBCASE("Transpose")
//         {
//             SUBCASE("float")
//             {
//                 std::cout << "Transpose - float" << std::endl;
//                 TestTranspose<float>(device);
//             }
//             SUBCASE("int")
//             {
//                 std::cout << "Transpose - int" << std::endl;
//                 TestTranspose<int>(device);
//             }
//         }
//     }
// }

TEST_CASE("GraphTest")
{
    // SUBCASE("SimpleGraph - ReLU")
    // {
    //     SimpleGraphTestReLU();
    // }
    //
    // SUBCASE("SimpleGraph - Sigmoid")
    // {
    //     SimpleGraphTestSigmoid();
    // }

    SUBCASE("MNIST - ReLU")
    {
        MnistTrainTest2();
    }
}

// TEST_CASE("ConcurrentCopy - small")
// {
//     //! Spawn 10 threads and copy SharedPtr
//     //! Check if reference counter successfully returns 1 at the end
//     ConcurrentCopy(10, 100);
// }
//
// TEST_CASE("ConcurrentCopy - large")
// {
//     //! Spawn 100 threads and copy SharedPtr
//     //!  Check if reference counter successfully returns 1 at the end
//     ConcurrentCopy(100, 100);
// }
//
// TEST_CASE("Weakptr - OwnershipTransfer")
// {
//     SimpleOwnershipTransfer();
// }
}
