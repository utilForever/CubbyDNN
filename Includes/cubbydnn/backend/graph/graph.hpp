
/**
 * Created by Justin on 18. 11. 13.
 *
 * @brief This file contains definitions of methods that adds operations to the graph
 */

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include "backend/graph_decl/graph_decl.hpp"
namespace cubby_dnn
{
template <typename T>
tensor<T> generate<T>::placeholder(const tensor_shape &shape, stream<T> &stream,
                                   const std::string &name)
{
    if (!shape::check_shape(shape, name))
    {
        return get_default_tensor();
    }

    long operation_id = operation_management<T>::get_next_operation_id();
    tensor<T> output_tensor(tensor_type::placeHolder, shape, operation_id);
    /// declare empty operation for later use
    auto new_op = placeholder_op<T>(operation_id, shape, stream, name);
    operation_management<T>::add_operation(new_op);
    return output_tensor;
}

template <typename T>
tensor <T> generate<T>::variable(const tensor_shape &shape, bool trainable, const std::string &name)
{
    if (!shape::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    long operation_id = operation_management<T>::get_next_operation_id();

    tensor<T> output_tensor(tensor_type ::variable, shape, operation_id);

    if (!trainable)
        output_tensor.make_constant();

    /// declare empty operation for later use
    auto new_op = weight_op<T>(operation_id, shape, name);
    operation_management<T>::add_operation(new_op);
    return output_tensor;
}

template <typename T>
tensor<T> operate<T>::mat_mul(tensor<T> &tensor1, tensor<T> &tensor2,
                              const std::string &name)
{
    std::vector<tensor<T>> tensor_vect;

    if (!tensor1.is_valid() || !tensor2.is_valid())
    {
        return get_default_tensor();
    }

    /// number of rows of first tensor should be identical to number of
    /// columns of second tensor
    if (tensor1.get_shape().cols() != tensor2.get_shape().rows() ||
        tensor1.get_shape().height() != tensor2.get_shape().height())
    {
        std::cout << "tensor shapes doesn't match for multiplication"
                  << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    tensor1.add_to(this_id);
    tensor2.add_to(this_id);
    // TODO: find way to initialize the default data

    auto tensor_object1 = tensor_object<T>(
        tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    auto tensor_object2 = tensor_object<T>(
        tensor2.get_data_size(),
        tensor2.get_shape(), tensor2.get_type(), tensor2.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();
    if (!tensor2.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    long tensor_object2_id = tensor_object_management<T>::add_tensor_object(tensor_object2);

    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);
    operation_management<T>::get_operation(tensor2.get_from()).add_output(tensor_object2_id);

    tensor_shape new_shape(tensor1.get_shape().rows(),
                           tensor2.get_shape().cols(),
                           tensor1.get_shape().height());

    tensor<T> output_tensor(tensor_type ::normal, new_shape, this_id);
    mat_mul_op<T> mat_mul_op(this_id, name);
    mat_mul_op.add_input(tensor_object1_id);
    mat_mul_op.add_input(tensor_object2_id);
    operation_management<T>::add_operation(mat_mul_op);
    return output_tensor;
}

template <typename T>
tensor<T> operate<T>::mad_add(tensor<T> &tensor1, tensor<T> &tensor2,
                              const std::string &name)
{
    if (!tensor1.is_valid() || !tensor2.is_valid())
    {
        return get_default_tensor();
    }

    ///check: both tensors should have identical shape
    if (tensor1.get_shape() != tensor2.get_shape())
    {
        std::cout << "tensor shapes doesn't match for Addition" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    tensor1.add_to(this_id);
    tensor2.add_to(this_id);
    // TODO: find way to initialize the default data

    auto tensor_object1 = tensor_object<T>(tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    auto tensor_object2 = tensor_object<T>(
        tensor2.get_data_size(),
        tensor2.get_shape(), tensor2.get_type(), tensor2.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();
    if (!tensor2.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    long tensor_object2_id = tensor_object_management<T>::add_tensor_object(tensor_object2);

    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);
    operation_management<T>::get_operation(tensor2.get_from()).add_output(tensor_object2_id);

    // setting the return tensor
    tensor_shape new_shape = tensor1.get_shape();
    // row size of the first tensor * col size of the second tensor

    tensor<T> output_tensor(tensor_type ::normal, new_shape, this_id);

    mat_add_op<T> mat_add_op(this_id, name);
    mat_add_op.add_input(tensor_object1_id);
    mat_add_op.add_input(tensor_object2_id);
    operation_management<T>::add_operation(mat_add_op);
    return output_tensor;
}

template <typename T>
tensor<T> operate<T>::mat_dot(tensor<T> &tensor1, T multiplier,
                              const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object1 = tensor_object<T>(
        tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);

    tensor_shape new_shape = tensor1.get_shape();

    tensor<T> output_tensor(tensor_type ::normal, new_shape, this_id);
    mat_dot_op<T> mat_dot_op(this_id, name, multiplier);
    mat_dot_op.add_input(tensor_object1_id);
    operation_management<T>::add_operation(mat_dot_op);
    return output_tensor;
}

template <typename T>
tensor<T> operate<T>::reshape(tensor<T> &tensor1, const tensor_shape &shape,
                              const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    if (!shape::check_shape(tensor1.get_shape(), name))
    {
        std::cout << "tensor shapes doesn't match for reshaping" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    if (tensor1.get_data_size() != shape.size())
    {
        std::cout << "size of new shape doesn't match for reshaping"
                  << std::endl;
        std::cout << "new size: " << std::to_string(shape.size())
                  << "original size: "
                  << std::to_string(tensor1.get_data_size()) << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data

    auto tensor_object1 = tensor_object<T>(
        tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);

    tensor_shape new_shape = shape;

    tensor<T> output_tensor(tensor_type ::normal, new_shape, this_id);
    reshape_op<T> reshape_op(this_id, name, shape);
    reshape_op.add_input(tensor_object1_id);
    operation_management<T>::add_operation(reshape_op);
    return output_tensor;
}

template <typename T>
tensor<T> operate<T>::one_hot(tensor<T> &tensor1, size_t size,
                              const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    if (!shape::check_shape(tensor1.get_shape(), name))
    {
        std::cout << "tensor shapes doesn't match for reshaping" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    // if reshaping size is different, make it false
    if (tensor1.get_data_size() != size)
    {
        std::cout << "size of new shape doesn't match for reshaping"
                  << std::endl;
        std::cout << "new size: " << std::to_string(size) << "original size: "
                  << std::to_string(tensor1.get_data_size()) << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    auto tensor_object1 = tensor_object<T>(
        tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);

    tensor_shape new_shape(size, 1, 1);

    tensor<T> output_tensor(tensor_type ::normal, new_shape, this_id);
    reshape_op<T> one_hot_op(this_id, name, new_shape);
    one_hot_op.add_input(tensor_object1_id);
    operation_management<T>::add_operation(one_hot_op);
    return output_tensor;
}

template <typename T>
void final<T>::wrapper(tensor<T> &tensor1, const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return;
    }

    auto this_id = operation_management<T>::get_next_operation_id();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data

    auto tensor_object1 = tensor_object<T>(
        tensor1.get_data_size(),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    if (!tensor1.is_mutable())
        tensor_object1.make_constant();

    long tensor_object1_id = tensor_object_management<T>::add_tensor_object(tensor_object1);
    operation_management<T>::get_operation(tensor1.get_from()).add_output(tensor_object1_id);

    wrapper_op<T> wrapper_op(this_id, name);
    wrapper_op.add_input(tensor_object1_id);
    operation_management<T>::add_operation(wrapper_op);
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATOR_TENSOR_HPP
