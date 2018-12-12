//
// Created by jwkim98 on 18. 12. 12.
//
#include <deque>
#include <mutex>
#include "graph_management.hpp"

#ifndef CUBBYDNN_PROCESS_MANAGEMENT_HPP
#define CUBBYDNN_PROCESS_MANAGEMENT_HPP

namespace cubby_dnn
{
/**
 *  @brief This class will manage process execution control
 */
class process_management
{
 public:
    /// pop next process to execute
    long pop_process();
    /// push the process which is ready to be executed
    void push_process(long operation_id);

    template <typename T>
    void increment_process_count_of(long operation_id)
    {
        operation<T>& operation =
            operation_management<T>::get_operation(operation_id);
        operation.increment_process_count();
        std::vector<long> output_tensor_vector =
            operation.get_output_tensor_vector();
        for (auto elem : output_tensor_vector)
        {
            tensor_object_management<T>::tensor_object_vector.at(elem)
                .increment_process_count();
        }
    }

 private:
    // TODO: think about way to check operations which are ready to be executed
    static std::deque<long> process_queue;
    static std::mutex lock_queue;
};

/// this queue stores operation_id of operations ready to be executed.
/// operations in this queue are allowed to be simultaneously executed
std::deque<long> process_management::process_queue;
}  // namespace cubby_dnn

#endif  // CUBBYDNN_PROCESS_MANAGEMENT_HPP
