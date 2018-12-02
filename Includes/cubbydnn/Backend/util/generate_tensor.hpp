//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include "Backend/operations/base_operations.hpp"
#include "Backend/util_decl/generate_tensor_decl.hpp"
#include "Backend/util_decl/shape_checker.hpp"

namespace cubby_dnn
{
template <typename T>
tensor<T> generate<T>::placeholder(const tensor_shape &shape, stream<T> &stream,
                                   const std::string &name)
{
    if (!shape_checker::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    auto operation_id = operation_management<T>::number_of_operations();
    tensor<T> rtn_tensor(tensor_type::placeHolder, shape,
                         static_cast<int>(operation_id), true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op = placeholder_op<T>(operation_id, stream, name);
    // add the operation to the global operation list
    operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> generate<T>::weight(const tensor_shape &shape, bool trainable,
                              const std::string &name)
{
    (void)trainable;

    if (!shape_checker::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    auto operation_id = operation_management<T>::number_of_operations();

    tensor<T> rtn_tensor(tensor_type ::weight, shape,
                         static_cast<int>(operation_id), true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op = weight_op<T>(operation_id, name);
    // add the operation to the global operation list
    operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> generate<T>::filter(const tensor_shape &shape, bool trainable,
                              const std::string &name)
{
    (void)trainable;

    if (!shape_checker::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    // adj_management<T>::add_op_adj();
    auto operation_id = operation_management<T>::number_of_operations();

    tensor<T> rtn_tensor(tensor_type ::filter, shape, operation_id, true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op = weight_op<T>(operation_id, name);
    // add the operation to the global operation list
    operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> operate<T>::mat_mul(tensor<T> &tensor1, tensor<T> &tensor2,
                              const std::string &name)
{
    // validity checking
    std::vector<tensor<T>> tensor_vect;

    if (!tensor1.is_valid() || !tensor2.is_valid())
    {
        return get_default_tensor();
    }

    if (tensor1.get_shape().cols() != tensor2.get_shape().rows() ||
        tensor1.get_shape().height() != tensor2.get_shape().height())
    {
        // number of rows of first tensor should be identical to number of
        // columns of second tensor
        std::cout << "tensor shapes doesn't match for multiplication"
                  << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::number_of_operations();

    tensor1.add_to(this_id);
    tensor2.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    auto tensor_object_ptr2 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor2.get_data_size())),
        tensor2.get_shape(), tensor2.get_type(), tensor2.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);
    tensor2.add_tensor_object(tensor_object_ptr2);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);
    operation_management<T>::add_output_of(tensor2.get_from(),
                                           tensor_object_ptr2);

    // setting the return tensor
    // numCols of first tensor, numRows of second tensor and dimension of third
    // tensor
    tensor_shape new_shape(tensor1.get_shape().rows(), tensor2.get_shape().cols(),
                    tensor1.get_shape().height());
    // row size of the first tensor * col size of the second tensor

    tensor<T> rtn_tensor(tensor_type ::normal, new_shape, this_id, true,
                         "tensor_from_op: " + name);
    mat_mul_op<T> mat_mul_op(this_id, name);
    mat_mul_op.add_input(tensor_object_ptr1);
    mat_mul_op.add_input(tensor_object_ptr2);
    operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> operate<T>::mad_add(tensor<T> &tensor1, tensor<T> &tensor2,
                              const std::string &name)
{
    if (!tensor1.is_valid() || !tensor2.is_valid())
    {
        return get_default_tensor();
    }

    if (tensor1.get_shape() != tensor2.get_shape())
    {
        std::cout << "tensor shapes doesn't match for Addition" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::number_of_operations();

    tensor1.add_to(this_id);
    tensor2.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    auto tensor_object_ptr2 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor2.get_data_size())),
        tensor2.get_shape(), tensor2.get_type(), tensor2.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);
    tensor2.add_tensor_object(tensor_object_ptr2);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);
    operation_management<T>::add_output_of(tensor2.get_from(),
                                           tensor_object_ptr2);

    // setting the return tensor
    tensor_shape new_shape = tensor1.get_shape();
    // row size of the first tensor * col size of the second tensor

    tensor<T> rtn_tensor(tensor_type ::normal, new_shape, this_id, true,
                         "tensor_from_op: " + name);

    mat_add_op<T> mat_add_op(this_id, name);
    mat_add_op.add_input(tensor_object_ptr1);
    mat_add_op.add_input(tensor_object_ptr2);
    operation_management<T>::add_op(mat_add_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> operate<T>::mat_dot(tensor<T> &tensor1, T multiplier,
                              const std::string &name)
{
    (void)multiplier;

    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    auto this_id = operation_management<T>::number_of_operations();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);

    // setting the return tensor
    tensor_shape new_shape = tensor1.get_shape();
    // row size of the first tensor * col size of the second tensor

    tensor<T> rtn_tensor(tensor_type ::normal, new_shape, this_id, true,
                         "tensor_from_op: " + name);
    mat_dot_op<T> mat_dot_op(this_id, name);
    mat_dot_op.add_input(tensor_object_ptr1);
    operation_management<T>::add_op(mat_dot_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> operate<T>::reshape(tensor<T> &tensor1, const tensor_shape &shape,
                              const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    if (!shape_checker::check_shape(tensor1.get_shape(), name))
    {
        std::cout << "tensor shapes doesn't match for reshaping" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    // if reshaping size is different, make it false
    if (tensor1.get_data_size() != shape.size())
    {
        std::cout << "size of new shape doesn't match for reshaping"
                  << std::endl;
        std::cout << "new size: " << std::to_string(shape.size())
                  << "original size: "
                  << std::to_string(tensor1.get_data_size()) << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
    }

    auto this_id = operation_management<T>::number_of_operations();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);

    // setting the return tensor
    tensor_shape new_shape = shape;
    // row size of the first tensor * col size of the second tensor

    tensor<T> rtn_tensor(tensor_type ::normal, new_shape, this_id, true,
                         "tensor_from_op: " + name);
    reshape_op<T> reshape_op(this_id, name);
    reshape_op.add_input(tensor_object_ptr1);
    operation_management<T>::add_op(reshape_op);
    return rtn_tensor;
}

template <typename T>
tensor<T> operate<T>::one_hot(tensor<T> &tensor1, unsigned long size,
                              const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return get_default_tensor();
    }

    if (!shape_checker::check_shape(tensor1.get_shape(), name))
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

    auto this_id = operation_management<T>::number_of_operations();

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);

    // setting the return tensor
    tensor_shape new_shape(size, 1, 1);
    // row size of the first tensor * col size of the second tensor

    tensor<T> rtn_tensor(tensor_type ::normal, new_shape, this_id, true,
                         "tensor_from_op: " + name);
    reshape_op<T> onehot_op(this_id, name);
    onehot_op.add_input(tensor_object_ptr1);
    operation_management<T>::add_op(onehot_op);
    return rtn_tensor;
}

template <typename T>
void Final<T>::wrapper(tensor<T> &tensor1, const std::string &name)
{
    if (!tensor1.is_valid())
    {
        return;
    }

    auto this_id = operation_management<T>::number_of_operations();

    tensor1.add_to(this_id);
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<tensor_object<T>>(
        std::vector<T>(static_cast<unsigned long>(tensor1.get_data_size())),
        tensor1.get_shape(), tensor1.get_type(), tensor1.get_from(), this_id);

    tensor1.add_tensor_object(tensor_object_ptr1);

    operation_management<T>::add_output_of(tensor1.get_from(),
                                           tensor_object_ptr1);

    wrapper_op<T> wrapper_op(this_id, name);
    wrapper_op.add_input(tensor_object_ptr1);
    operation_management<T>::add_op(wrapper_op);
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATOR_TENSOR_HPP
