#include <Test.h>
#include <iostream>
#include "Backend/util/generate_tensor.hpp"

using namespace cubby_dnn;

std::vector<std::tuple<long, unsigned long, unsigned long, std::string>> Example3()
{
    File_stream<float> file_stream;
    auto input_tensor1 = Generate<float>::placeHolder(
        Shape(5, 4, 3), file_stream, "test placeHolder operation1");
    auto input_tensor2 =
        Generate<float>::weight(Shape(4, 5, 3), true, "test weight operation1");

    auto input_tensor3 =
        Generate<float>::weight(Shape(5, 5, 3), true, "test weight operation2");

    auto multiplied_tensor1 = Operate<float>::matMul(
        input_tensor1, input_tensor2, "test matMul operation1");  // shape: 5,5,3

    auto multiplied_tensor2 =
        Operate<float>::matMul(multiplied_tensor1, input_tensor3,
                               "test matMul operation2");  // shape: 5,5,3

    auto added_tensor1 =
        Operate<float>::matAdd(multiplied_tensor2, input_tensor3,
                               "test matMul operation3");  // shape: 5,5,3

    auto reshaped_tensor1 = Operate<float>::reshape(
        added_tensor1, Shape(75, 1, 1), "test reshape operation1");

    Final<float>::wrapper(reshaped_tensor1, "test wrapper operation1");

    return Operation_management<float>::get_operation_info();
}

std::vector<std::tuple<long, unsigned long, unsigned long, std::string>> Example2()
{
    File_stream<int> file_stream;
    auto input_tensor1 = Generate<int>::placeHolder(
        Shape(2, 4, 3), file_stream, "test placeHolder operation1");  // 1

    auto input_tensor2 = Generate<int>::weight(Shape(4, 2, 3), true,
                                               "test weight operation2");  // 2

    auto multiplied_tensor1 =
        Operate<int>::matMul(input_tensor2, input_tensor1,
                             "test matMul operation1");  // 3
    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation1"); //4

    return Operation_management<int>::get_operation_info();
}

std::vector<std::tuple<long, unsigned long, unsigned long, std::string>> Example1()
{
    File_stream<int> file_stream;
    auto input_tensor1 = Generate<int>::placeHolder(
        Shape(2, 2, 1), file_stream, "test placeHolder operation1");  // 1

    auto input_tensor2 = Generate<int>::weight(Shape(2, 2, 1), true,
                                               "test weight operation2");  // 2

    auto multiplied_tensor1 =
        Operate<int>::matMul(input_tensor2, input_tensor1,
                             "test matMul operation1");  // 3

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation1");  // 4

    auto multiplied_tensor2 =
        Operate<int>::matMul(input_tensor2, input_tensor1,
                             "test matMul operation2");  // 5

    auto added_tensor1 = Operate<int>::matAdd(input_tensor1, input_tensor2,
                                              "test matAdd operation1");  // 6

    auto dot_operated_tensor1 =
        Operate<int>::matDot(added_tensor1, 5, "test matDot operation1");  // 7

    auto reshaped_tensor1 = Operate<int>::reshape(
        dot_operated_tensor1, Shape(1, 2, 2), "test reshape operation1");  // 8

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation2");  // 9

    Final<int>::wrapper(multiplied_tensor2, "test wrapper operation3");  // 10

    Final<int>::wrapper(reshaped_tensor1, "test wrapper operation4");  // 11

    Operation_management<int>::print_operation_info();

    Operation_management<int>::create_adj();

    Adj_management<int>::print_adj();

    return Operation_management<int>::get_operation_info();
}

int Add(int a, int b)
{
    // Example of multiplying two tensors of shape {1,1,1}
    return a + b;
}
