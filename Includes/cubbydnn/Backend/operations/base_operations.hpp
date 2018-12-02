//
// Created by Justin on 18. 11. 16.
//

#ifndef CUBBYDNN_BASE_OPERATIONS_HPP
#define CUBBYDNN_BASE_OPERATIONS_HPP
#include "Backend/operations_decl/base_operations_decl.hpp"
namespace cubby_dnn
{
template <typename T>
operation<T>::operation() = default;

template <typename T>
mat_mul_op<T>::mat_mul_op(std::shared_ptr<tensor_object<T>> tensor1,
                          std::shared_ptr<tensor_object<T>> tensor2,
                          std::shared_ptr<tensor_object<T>> output_tensor,
                          unsigned long operation_id, const std::string &name)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(tensor2);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->operation_id = operation_id;
}

template <typename T>
mat_mul_op<T>::mat_mul_op(
    std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect,
    std::shared_ptr<tensor_object<T>> output_tensor, unsigned long operation_id,
    const std::string &name)
{
    for (std::shared_ptr<tensor_object<T>> ptr : tensor_vect)
    {
        this->input_tensor_vect.emplace_back(ptr);
    }
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
mat_add_op<T>::mat_add_op(std::shared_ptr<tensor_object<T>> tensor1,
                          std::shared_ptr<tensor_object<T>> tensor2,
                          std::shared_ptr<tensor_object<T>> output_tensor,
                          unsigned long operation_id, const std::string &name)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(tensor2);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
mat_add_op<T>::mat_add_op(
    std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect,
    std::shared_ptr<tensor_object<T>> output_tensor, unsigned long operation_id,
    const std::string &name)
{
    for (std::shared_ptr<tensor_object<T>> ptr : tensor_vect)
    {
        this->input_tensor_vect.emplace_back(ptr);
    }
    this->outpt_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
mat_dot_op<T>::mat_dot_op(std::shared_ptr<tensor_object<T>> tensor1,
                          std::shared_ptr<tensor_object<T>> identity_tensor,
                          std::shared_ptr<tensor_object<T>> output_tensor,
                          unsigned long operation_id, const std::string &name)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(identity_tensor);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
reshape_op<T>::reshape_op(std::shared_ptr<tensor_object<T>> tensor1,
                          std::shared_ptr<tensor_object<T>> output_tensor,
                          const tensor_shape &shape,
                          unsigned long operation_id, const std::string &name)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::middle);
    this->shape = shape;
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
placeholder_op<T>::placeholder_op(
        std::shared_ptr<tensor_object<T>> output_tensor,
        const tensor_shape &shape, unsigned long operation_id,
        const std::string &name)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::start);
    this->shape = shape;
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
weight_op<T>::weight_op(std::shared_ptr<tensor_object<T>> output_tensor,
                        const tensor_shape &shape,
                        unsigned long operation_id, const std::string &name)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::start);
    this->shape = shape;
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
constant_op<T>::constant_op(std::shared_ptr<tensor_object<T>> output_tensor,
                            const tensor_shape &shape,
                            unsigned long operation_id, const std::string &name)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->op_type(operation_type::start);
    this->shape = shape;
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
wrapper_op<T>::wrapper_op(std::shared_ptr<tensor_object<T>> input_tensor,
                          unsigned long operation_id, const std::string &name)
{
    this->input_tensor_vect.emplace_back(input_tensor);
    this->op_type = operation_type::final;
    this->name = name;
    this->operation_id = operation_id;
}

template <typename T>
void operation_management<T>::add_op(operation<T> operation)
{
    std::lock_guard<std::mutex> guard(operation_list_mutex);
    operation_list.emplace_back(operation);
}

template <typename T>
void operation_management<T>::set_op(unsigned int id,
                                     const operation<T> &operation)
{
    std::lock_guard<std::mutex> guard(operation_list_mutex);
    operation_list[id] = operation;
}

template <typename T>
void operation_management<T>::add_output_of(
    long id, std::shared_ptr<tensor_object<T>> tensor_ptr)
{
    decltype(auto) operation = get_op(id);
    operation.add_output(tensor_ptr);
}

template <typename T>
void operation_management<T>::add_input_of(
    long id, std::shared_ptr<tensor_object<T>> tensor_ptr)
{
    decltype(auto) operation = get_op(id);
    operation.add_input(tensor_ptr);
}
}  // namespace cubby_dnn
#endif  // CUBBYDNN_BASE_OPERATIONS_HPP
