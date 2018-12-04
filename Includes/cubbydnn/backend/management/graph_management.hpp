//
// Created by Justin on 18. 12. 4.
//

#ifndef CUBBYDNN_GRAPH_MANAGEMENT_HPP
#define CUBBYDNN_GRAPH_MANAGEMENT_HPP

#include <mutex>
#include "backend/graph/tensor.hpp"

namespace cubby_dnn
{
template <typename T>
class management
{
 public:
    static long add_tensor_object(const tensor_object<T> object)
    {
        tensor_object_vector.emplace_back(object);
        long tensor_object_id = static_cast<long>(tensor_object_vector.size());
        return tensor_object_id;
    }

    static tensor_object<T>& get_tensor_object(long index)
    {
        return tensor_object_vector.at(static_cast<size_t>(index));
    }

 private:
    static std::vector<tensor_object<T>> tensor_object_vector;
    static std::mutex tensor_object_vector_mutex;
};

template <typename T>
std::vector<tensor_object<T>> management<T>::tensor_object_vector;

template <typename T>
std::mutex management<T>::tensor_object_vector_mutex;
};  // namespace cubby_dnn

#endif  // CUBBYDNN_GRAPH_MANAGEMENT_HPP
