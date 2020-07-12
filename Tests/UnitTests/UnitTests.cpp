#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "SimpleTest.hpp"
#include "FunctionTest/NativeTest.hpp"
#include "FunctionTest/BlazeTest.hpp"
#include "UtilTests/SharedPtrTests.hpp"
#include "UtilTests/WeakPtrTests.hpp"
#include "GraphTest/SimpleMnist.hpp"
#include <doctest.h>
#include <iostream>

using namespace CubbyDNN::Test;

TEST_CASE("Simple test")
{
    CHECK(5 == Add(2, 3));
    std::cout << "called Test" << std::endl;
}

// TEST_CASE("SimpleMnist")
// {
//     SimpleMnistTest();
// }

// TEST_CASE("Simple matmul 1")
// {
//     TestMatMul();
// }
//
// TEST_CASE("Simple matmul 2")
// {
//     TestMatMul2();
// }
//
// TEST_CASE("MatAdd")
// {
//     TestMatAdd();
// }
//
// TEST_CASE("Matdot")
// {
//     TestMatDot();
// }
//
//
// TEST_CASE("Blaze simpleMatmul1")
// {
//     TestBlazeMul();
// }
//
// TEST_CASE("Blaze simpleMatmul2")
// {
//     TestBlazeMul2();
// }
//
// TEST_CASE("Blaze simpleMatAdd")
// {
//     TestBlazeAdd();
// }
//
// TEST_CASE("Blaze MatDot")
// {
//     TestBlazeDot();
// }
//
//
// TEST(TaskQueueTest, UtilTest)
// {
//     TaskQueueTest(12);
// }

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
