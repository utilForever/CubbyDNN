#include <gtest/gtest.h>
#include <SimpleTest.hpp>

using namespace cubby_dnn;

std::vector<operation_info> Example1()
{

    ///Initialize every graph management classes before use (not required in real use)
    ///prevents graph conflicts between tests
    operation_management<int>::clear();
    adjacency_management<int>::clear();
    tensor_object_management<int>::clear();

    /// file stream class template is used for streaming data inside external
    /// files
    file_stream<int> file_stream;

    /// operation that streams file data into tensor in shape (2,2,1)
    auto input_tensor1 =
            generate<int>::placeholder(tensor_shape(2, 2, 1), file_stream);

    /// operation that returns tensor variable in shape (2, 2, 1), and mark it
    /// trainable(non- constant)
    auto input_tensor2 = generate<int>::variable(tensor_shape(2, 2, 1));

    /// operation that multiplies two tensors and returns tensor to contain the
    /// result
    auto multiplied_tensor1 =
            operate<int>::mat_mul(input_tensor1, input_tensor2);

    /// operation that receives tensor and marks given tensor has no more
    /// further operation (end of the graph)
    final<int>::wrapper(multiplied_tensor1);

    auto multiplied_tensor2 =
            operate<int>::mat_mul(input_tensor1, input_tensor2);

    auto added_tensor1 = operate<int>::mat_add(input_tensor1, input_tensor2);

    /// operation that calculates dot-product of given tensor and multiplier and
    /// returns tensor to contain the result
    auto dot_operated_tensor1 = operate<int>::mat_dot(added_tensor1, 5);

    /// operation that reshapes given tensor to another shape (total size of
    /// given tensor, and return shape must match)
    auto reshaped_tensor1 =
            operate<int>::reshape(dot_operated_tensor1, tensor_shape(1, 2, 2));

    final<int>::wrapper(multiplied_tensor1);

    final<int>::wrapper(multiplied_tensor2);

    final<int>::wrapper(reshaped_tensor1);

    operation_management<int>::print_operation_info();

    adjacency_management<int>::print_adjacency_matrix();

    return operation_management<int>::get_operation_info();
}

std::vector<operation_info> Example2()
{
    operation_management<int>::clear();
    adjacency_management<int>::clear();
    tensor_object_management<int>::clear();

    file_stream<int> file_stream;
    auto input_tensor1 =
            generate<int>::placeholder(tensor_shape(2, 4, 3), file_stream);

    auto input_tensor2 = generate<int>::variable(tensor_shape(4, 2, 3));

    auto multiplied_tensor1 =
            operate<int>::mat_mul(input_tensor2, input_tensor1);

    final<int>::wrapper(multiplied_tensor1);

    operation_management<int>::print_operation_info();

    adjacency_management<int>::print_adjacency_matrix();

    return operation_management<int>::get_operation_info();
}

std::vector<operation_info> Example3()
{
    operation_management<int>::clear();
    adjacency_management<int>::clear();
    tensor_object_management<int>::clear();

    file_stream<float> file_stream;
    auto input_tensor1 =
            generate<float>::placeholder(tensor_shape(5, 4, 3), file_stream);
    auto input_tensor2 = generate<float>::variable(tensor_shape(4, 5, 3));

    auto input_tensor3 = generate<float>::variable(tensor_shape(5, 5, 3));

    auto multiplied_tensor1 =
            operate<float>::mat_mul(input_tensor1, input_tensor2);  // shape: 5,5,3

    auto multiplied_tensor2 = operate<float>::mat_mul(
            multiplied_tensor1, input_tensor3);  // shape: 5,5,3

    auto added_tensor1 = operate<float>::mat_add(
            multiplied_tensor2, input_tensor3);  // shape: 5,5,3

    auto reshaped_tensor1 =
            operate<float>::reshape(added_tensor1, tensor_shape(75, 1, 1));

    final<float>::wrapper(reshaped_tensor1);

    operation_management<float>::print_operation_info();

    adjacency_management<float>::print_adjacency_matrix();

    return operation_management<float>::get_operation_info();
}

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
    std::cout << "called Test" << std::endl;
}

TEST(Test1, Example1)
{
    std::vector<cubby_dnn::operation_info> ans{
            operation_info{0, 0, 3, "placeholder"},
            operation_info{1, 0, 3, "weight"},
            operation_info{2, 2, 2, "mat_mul"},
            operation_info{3, 1, 0, "wrapper"},
            operation_info{4, 2, 1, "mat_mul"},
            operation_info{5, 2, 1, "mat_add"},
            operation_info{6, 1, 1, "mat_dot"},
            operation_info{7, 1, 1, "reshape"},
            operation_info{8, 1, 0, "wrapper"},
            operation_info{9, 1, 0, "wrapper"},
            operation_info{10, 1, 0, "wrapper"}
    };

    EXPECT_EQ(ans, Example1());
}

TEST(Test2, Example2)
{
    std::vector<cubby_dnn::operation_info> ans{
            operation_info{0, 0, 1, "placeholder"},
            operation_info{1, 0, 1, "weight"},
            operation_info{2, 2, 1, "mat_mul"},
            operation_info{3, 1, 0, "wrapper"}
    };

    EXPECT_EQ(ans, Example2());
}

TEST(Test3, Example3)
{
    std::vector<cubby_dnn::operation_info> ans{
            operation_info{0, 0, 1, "placeholder"},
            operation_info{1, 0, 1, "weight"},
            operation_info{2, 0, 2, "weight"},
            operation_info{3, 2, 1, "mat_mul"},
            operation_info{4, 2, 1, "mat_mul"},
            operation_info{5, 2, 1, "mat_add"},
            operation_info{6, 1, 1, "reshape"},
            operation_info{7, 1, 0, "wrapper"}
    };

    EXPECT_EQ(ans, Example3());
}
