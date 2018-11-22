#include <Test.h>
#include <iostream>
#include "Backend/util/generate_tensor.hpp"


using namespace cubby_dnn;

void simple_Example()
{
    File_stream<int> file_stream;
    auto input_tensor1 = Generate<int>::placeHolder(
            std::vector<int>{ 2, 2, 1 }, file_stream, "test placeHolder operation");


    auto input_tensor2 = Generate<int>::weight(std::vector<int>{ 2, 2, 1 },
                                               true, "test weight operation2");

    auto multiplied_tensor1 = Operate<int>::matMul(input_tensor2, input_tensor1,
                                                   "test matMul operation1");

    auto multiplied_tensor2 = Operate<int>::matMul(input_tensor2, input_tensor1,
                                                   "test matMul operation2");

    auto added_tensor1 = Operate<int>::matAdd(input_tensor1, input_tensor2, "test matAdd operation1");

    auto dot_operated_tensor1 = Operate<int>::matDot(input_tensor1, 5, "test matDot operation");

    auto reshaped_tensor1 = Operate<int>::reshape(input_tensor2, std::vector<int>{1, 4, 1}, "test reshape operation1");

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation1");

    Final<int>::wrapper(multiplied_tensor2, "test wrapper operation2");

    Operation_management<int>::get_operation_infos();
}
int Add(int a, int b)
{
    // Example of multiplying two tensors of shape {1,1,1}
    simple_Example();
    return a + b;
}


