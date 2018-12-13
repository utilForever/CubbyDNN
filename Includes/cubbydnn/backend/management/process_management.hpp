//
// Created by jwkim98 on 18. 12. 13.
//

#include <backend/management_decl/process_management_decl.hpp>

#include "backend/management_decl/process_management_decl.hpp"

#ifndef CUBBYDNN_PROCESS_MANAGEMENT_HPP
#define CUBBYDNN_PROCESS_MANAGEMENT_HPP

namespace cubby_dnn
{
template <typename T>
long process_management<T>::get_process()
{
    if (operation_management<T>::operation_vector_size() <
        process_management<T>::minimum_queue_size)
        update_available_process();
    std::lock_guard<std::mutex> lock(lock_queue);
    long operation_id = process_queue.front();
    process_queue.pop_front();
    return operation_id;
}

template <typename T>
void process_management<T>::increment_process_count_of(long operation_id)
{
    operation<T>& operation =
        operation_management<T>::get_operation(operation_id);
    operation.increment_process_count();
    std::vector<long> output_tensor_vector =
        operation.get_output_tensor_vector();

    for (auto tensor_id : output_tensor_vector)
    {
        tensor_object_management<T>::tensor_object_vector.at(tensor_id)
            .increment_process_count();
    }
}

template <typename T>
std::unique_ptr<typename tensor_object<T>::data>
process_management<T>::get_tensor_data_ptr(long tensor_id)
{
    return std::move(
        tensor_object_management<T>::get_tensor_data_ptr(tensor_id));
}

template <typename T>
void process_management<T>::return_tensor_data_ptr(
    long tensor_id, std::unique_ptr<typename tensor_object<T>::data> rhs)
{
    tensor_object_management<T>::return_tensor_data_ptr(std::move(rhs));
}

template <typename T>
void process_management<T>::push_process(long operation_id)
{
    std::lock_guard<std::mutex> lock(lock_queue);
    process_queue.push_back(operation_id);
}

template <typename T>
void process_management<T>::update_available_process()
{
    for (auto count = 0;
         count < operation_management<T>::operation_vector_size(); count++)
    {
        if (operation_management<T>::check_available(count))
        {
            push_process(count);
        }
    }
}

}  // namespace cubby_dnn

#endif  // CUBBYDNN_PROCESS_MANAGEMENT_HPP
