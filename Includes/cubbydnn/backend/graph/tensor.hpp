//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
#define CUBBYDNN_TENSOR_CONTAINER_DEF_HPP

#include <algorithm>
#include <functional>
#include "backend/graph_decl/tensor_decl.hpp"

namespace cubby_dnn
{
/// definitions

template <typename T>
tensor<T>::tensor(tensor_type type, const tensor_shape &shape, long from,
                  bool _mutable)
    : from(from), _mutable(_mutable), type(type), shape(shape)
{
}


template <typename T>
struct tensor_object<T>::storage
{
 public:
    storage(const std::vector<T> &data, const tensor_shape &shape);

    storage(tensor_shape &&data, tensor_shape &&shape);

    std::vector<T> data;  // stores actual data with data type 'T'
    tensor_shape shape;          // shape of the data
    typedef std::size_t size_type;
    size_type byte_size;
};

template <typename T>
tensor_object<T>::storage::storage(const std::vector<T> &data,
                                   const tensor_shape &shape)
{
    this->data = data;
    this->shape = shape;
    byte_size = this->data.size();
}

template <typename T>
tensor_object<T>::storage::storage(tensor_shape &&data, tensor_shape &&shape)
{
    this->data = std::forward<std::vector<T>>(data);
    this->shape = std::forward<tensor_shape>(data);
    byte_size = this->data.size();
}

template <typename T>
tensor_object<T>::tensor_object(const std::vector<T> &data, const tensor_shape &shape,
                                tensor_type type, long from, long to)
    : type(type), from(from), to(to)
{
    verify<T>(data, shape);

    this->tensor_storage = std::make_unique<storage>(data, shape);
}

template <typename T>
tensor_object<T>::tensor_object(std::vector<T> &&data, tensor_shape &&shape,
                                tensor_type type, long from, long to)
    : type(type), from(from), to(to)
{
    verify<T>(data, shape);  // checks exception if arguments are invalid

    this->tensor_storage = std::make_unique<storage>(
        std::forward<std::vector<T>>(data), std::forward<tensor_shape>(shape));
}

template <typename T>
tensor_object<T>::tensor_object(const tensor_object<T> &rhs)
    : tensor_storage(nullptr)
{
    if (!rhs.tensor_storage)
        this->tensor_storage = std::make_unique<tensor_object<T>::tensor_storage>(
            *rhs.tensor_storage);
}

template <typename T>
tensor_object<T>::tensor_object(tensor_object<T> &&rhs) noexcept = default;

template <typename T>
tensor_object<T> &tensor_object<T>::operator=(
    const cubby_dnn::tensor_object<T> &rhs)
{
    // may throw std::bad_alloc() (this function will provide strong guarantee)
    if (rhs.object)
        this->tensor_storage = std::make_unique<tensor_object<T>::tensor_storage>(
            *rhs.tensor_storage);
    return *this;
}

template <typename T>
tensor_object<T> &tensor_object<T>::operator=(
    cubby_dnn::tensor_object<T> &&rhs) noexcept = default;

template <typename T>
tensor_object<T>::~tensor_object() = default;

template <typename T>
bool verify(const std::vector<T> &data, const tensor_shape &shape)
{
    if (data.empty())
    {
        std::cout << "empty data" << std::endl;
        return false;
    }

    if (data.size() != shape.size())
    {
        std::string err_message = "data shape doesn't match";
        err_message += "Expected Size = " + std::to_string(shape.size());
        err_message += "given data size = " + std::to_string(data.size());
        std::cout << err_message << std::endl;
        return false;
    }
    return true;
}

// getters

template <typename T>
size_t tensor_object<T>::get_data_size() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_object is empty" << std::endl;
        return error_id;
    }
    return tensor_storage->data.size();
}

template <typename T>
size_t tensor_object<T>::get_data_byte_size() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_object is empty" << std::endl;
        return error_id;
    }
    return tensor_storage->data.size() * sizeof(T);
}

template <typename T>
const std::vector<T> &tensor_object<T>::get_data() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_object is empty" << std::endl;
        return std::vector<T>();
    }
    return tensor_storage->data;
}

/// management

template <typename T>
long adj_management<T>::add_op_adj()
{
    auto graph_size = adj_forward.size();

    auto expected_row_size = (graph_size + 1 > default_graph_size)
                                 ? graph_size + 1
                                 : default_graph_size;

    std::deque<std::shared_ptr<tensor_object<T>>> temp(expected_row_size,
                                                       nullptr);

    auto emplace_until_size =
        [expected_row_size, graph_size](
            std::deque<std::shared_ptr<tensor_object<T>>> &arg) mutable {
            while (expected_row_size > arg.size())
            {
                arg.emplace_back(nullptr);  // copy-construct new temp
            }
        };

    std::lock_guard<std::mutex> guard(adj_mutex);  // lock the adj_matrix

    if (expected_row_size > default_graph_size)
    {
        // for all rows, increment
        std::for_each(adj_forward.begin(), adj_forward.end(),
                      emplace_until_size);
    }

    adj_forward.emplace_back(temp);  // graph_size += 1

    return static_cast<long>(adj_forward.size());
}

template <typename T>
void adj_management<T>::add_edge(
    long from, long to, std::shared_ptr<tensor_object<T>> &tensor_object_ptr)
{
    auto graph_size = static_cast<long>(adj_forward.size());
    if (from == to)
    {
        std::string error_msg = "Operation may not connect to itself";
        std::cout << error_msg << std::endl;
    }

    if (graph_size + 1 < from || graph_size + 1 < to)
    {
        std::string error_msg = "pointing to operation that doesn't exist";
        error_msg += "graph size: " + std::to_string(adj_forward.size()) +
                     "from: " + std::to_string(from) +
                     "to: " + std::to_string(to);
        std::cout << error_msg << std::endl;
    }

    if (adj_forward[from][to] != nullptr)
    {
        std::string error_msg = "this edge is already assigned";
        std::cout << error_msg << std::endl;
    }

    std::lock_guard<std::mutex> guard(adj_mutex);
    adj_forward[from][to] = tensor_object_ptr;
}

template <typename T>
std::shared_ptr<tensor_object<T>> adj_management<T>::get_tensor_ptr(int from,
                                                                    int to)
{
    if (from >= adj_forward.size() || to >= adj_forward.size())
    {
        std::string error_msg = "pointing to operation that doesn't exist";
        error_msg +=
            ("graph size: " + std::to_string(adj_forward.size()) +
             "from: " + std::to_string(from) + "to: " + std::to_string(to));
        std::cout << error_msg << std::endl;
    }
    return adj_forward[from][to];  /// get ownership from adj (thread-safe);
}
}  // namespace cubby_dnn
#endif  // CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
