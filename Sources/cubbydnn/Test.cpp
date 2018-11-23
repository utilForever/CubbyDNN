#include <Test.h>
#include <iostream>
#include "Backend/util/generate_tensor.hpp"


using namespace cubby_dnn;

std::vector<std::tuple<long, unsigned long, unsigned long>> Example1()
{
    File_stream<int> file_stream;
    auto input_tensor1 = Generate<int>::placeHolder(
            std::vector<int>{ 2, 2, 1 }, file_stream, "test placeHolder operation"); //1

    auto input_tensor2 = Generate<int>::weight(std::vector<int>{ 2, 2, 1 },
                                               true, "test weight operation2"); //2

    auto multiplied_tensor1 = Operate<int>::matMul(input_tensor2, input_tensor1,
                                                   "test matMul operation1"); //3

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation0"); //4


    auto multiplied_tensor2 = Operate<int>::matMul(input_tensor2, input_tensor1,
                                                   "test matMul operation2"); //5

    auto added_tensor1 = Operate<int>::matAdd(input_tensor1, input_tensor2, "test matAdd operation1"); //6

    auto dot_operated_tensor1 = Operate<int>::matDot(added_tensor1, 5, "test matDot operation"); //7

    auto reshaped_tensor1 = Operate<int>::reshape(dot_operated_tensor1, std::vector<int>{1, 4, 1}, "test reshape operation1"); //8

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation1"); //9

    Final<int>::wrapper(multiplied_tensor2, "test wrapper operation2"); //10

    Final<int>::wrapper(reshaped_tensor1, "test wrapper operation3"); //11

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


