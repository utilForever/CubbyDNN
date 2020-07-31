#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "SimpleTest.hpp"
#include "FunctionTest/NativeTest.hpp"
#include "FunctionTest/BlazeTest.hpp"
#include "UtilTests/SharedPtrTests.hpp"
#include "UtilTests/WeakPtrTests.hpp"
#include "GraphTest/SimpleMnist.hpp"
#include "UtilTests/TensorTest.hpp"
#include <doctest.h>
#include <iostream>

using namespace CubbyDNN::Test;

TEST_CASE("Simple test")
{
    CHECK(5 == Add(2, 3));
    std::cout << "called Test" << std::endl;
}

TEST_CASE("SimpleMnist")
{
    SimpleMnistTest();
}

TEST_CASE("Tensor test")
{
    SUBCASE("Tensor Copy test")
    {
        TensorCopyTest();
    }
    SUBCASE("Tensor Move test")
    {
        TensorMoveTest();
    }
    SUBCASE("Tensor forward by copy")
    {
        TensorForwardTestWithCopy();
    }
    SUBCASE("Tensor forward by move")
    {
        TensorForwardTestWithMove();
    }
}

TEST_CASE("Native matrix Multiplication")
{
    SUBCASE("MatMul1")
    {
        TestMatMul();
    }
    SUBCASE("MatMul2")
    {
        TestMatMul2();
    }
    SUBCASE("Matmul with transpose")
    {
        TestMatMulWithTranspose();
    }
}

TEST_CASE("Native matix addition")
{
    TestMatAdd();
}

TEST_CASE("Matdot")
{
    TestMatDot();
}

TEST_CASE("TestShrink")
{
    SUBCASE("Shrink1")
    {
        TestShrink();
    }
    SUBCASE("Shrink2")
    {
        TestShrink2();
    }
}

TEST_CASE("TestScalarMul")
{
    SUBCASE("ScalarMul")
    {
        TestScalarMul();
    }
}


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
