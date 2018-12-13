//
// Created by jwkim98 on 18. 12. 12.
//

#include <deque>
#include <mutex>
#include "backend/management/graph_management.hpp"

#ifndef CUBBYDNN_PROCESS_MANAGEMENT_DECL_HPP
#define CUBBYDNN_PROCESS_MANAGEMENT_DECL_HPP

namespace cubby_dnn
{
/**
 *  @brief This class will manage process execution control
 */
template <typename T>
class process_management
{
 public:
    /// pop next process to execute
    /// @return operation_id that is available
    long get_process();

    /// this method must be called to allow the operation to execute next
    /// tensor_object
    void increment_process_count_of(long operation_id);

    /// searches for available operation
    /// @return operation_id which is ready to be executed

 private:
    // TODO: think about way to check operations which are ready to be executed
    /// push the process which is ready to be executed
    void push_process(long operation_id);

    void update_available_process();

    static std::deque<long> process_queue;
    static std::mutex lock_queue;
    static const long minimum_queue_size;
};

/// this queue stores operation_id of operations ready to be executed.
/// operations in this queue are allowed to be simultaneously executed
template <typename T>
std::deque<long> process_management<T>::process_queue;

template <typename T>
std::mutex process_management<T>::lock_queue;

template <typename T>
const long process_management<T>::minimum_queue_size = 3;

}  // namespace cubby_dnn

#endif  // CUBBYDNN_PROCESS_MANAGEMENT_HPP
