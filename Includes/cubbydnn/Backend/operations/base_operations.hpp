//
// Created by jwkim on 18. 11. 16.
//

#ifndef CUBBYDNN_BASE_OPERATIONS_HPP
#define CUBBYDNN_BASE_OPERATIONS_HPP
#include "Backend/operations_decl/base_operations_decl.hpp"
namespace cubby_dnn
{
template <typename T>
Operation<T>::Operation() = default;

template <typename T>
Mat_mul_op<T>::Mat_mul_op(std::shared_ptr<Tensor<T>> tensor1,
                          std::shared_ptr<Tensor<T>> tensor2,
                          std::shared_ptr<Tensor<T>> output_tensor)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(tensor2);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
}

template <typename T>
Mat_mul_op<T>::Mat_mul_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect,
                          std::shared_ptr<Tensor<T>> output_tensor)
{
    for (std::shared_ptr<Tensor<T>> ptr : tensor_vect)
    {
        this->input_tensor_vect.emplace_back(ptr);
    }
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
}

template <typename T>
Mat_add_op<T>::Mat_add_op(std::shared_ptr<Tensor<T>> tensor1,
                          std::shared_ptr<Tensor<T>> tensor2,
                          std::shared_ptr<Tensor<T>> output_tensor)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(tensor2);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
}

template <typename T>
Mat_add_op<T>::Mat_add_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect,
                          std::shared_ptr<Tensor<T>> output_tensor)
{
    for (std::shared_ptr<Tensor<T>> ptr : tensor_vect)
    {
        this->input_tensor_vect.emplace_back(ptr);
    }
    this->outpt_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
}

template <typename T>
Mat_dot_op<T>::Mat_dot_op(std::shared_ptr<Tensor<T>> tensor1,
                          std::shared_ptr<Tensor<T>> identity_tensor,
                          std::shared_ptr<Tensor<T>> output_tensor)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->input_tensor_vect.emplace_back(identity_tensor);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
}

template <typename T>
Reshape_op<T>::Reshape_op(std::shared_ptr<Tensor<T>> tensor1,
                          std::shared_ptr<Tensor<T>> output_tensor,
                          const std::vector<int> &shape)
{
    this->input_tensor_vect.emplace_back(tensor1);
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::basic);
    this->shape = shape;
}

template <typename T>
placeHolder_op<T>::placeHolder_op(std::shared_ptr<Tensor<T>> output_tensor,
                                  const std::vector<int> &shape)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::generate);
    this->shape = shape;
}

template <typename T>
weight_op<T>::weight_op(std::shared_ptr<Tensor<T>> output_tensor,
                        const std::vector<int> &shape)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::generate);
    this->shape = shape;
}

template <typename T>
constant_op<T>::constant_op(std::shared_ptr<Tensor<T>> output_tensor,
                            const std::vector<int> &shape)
{
    this->output_tensor_vect.emplace_back(output_tensor);
    this->operation_type(operation_type::generate);
    this->shape = shape;
}

template <typename T>
void Operation_management<T>::add_op(const Operation <T> &operation, unsigned long id)
{
    std::lock_guard<std::mutex> guard(operation_list_mutex);
    operation_list.emplace_back();
    operation.operation_id = id;
}

}  // namespace cubby_dnn

#endif  // CUBBYDNN_BASE_OPERATIONS_HPP
