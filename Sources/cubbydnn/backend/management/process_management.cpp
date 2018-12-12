//
// Created by jwkim98 on 18. 12. 12.
//
#include <backend/management/process_management.hpp>

#include "backend/management/process_management.hpp"

namespace cubby_dnn
{
long process_management::pop_process()
{
    std::lock_guard<std::mutex> lock(lock_queue);
    long temp = process_queue.front();
    process_queue.pop_front();
    return temp;
}

void process_management::push_process(long operation_id)
{
    std::lock_guard<std::mutex> lock(lock_queue);
    process_queue.push_back(operation_id);
}

}  // namespace cubby_dnn