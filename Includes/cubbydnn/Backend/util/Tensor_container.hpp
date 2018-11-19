//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
#define CUBBYDNN_TENSOR_CONTAINER_DEF_HPP

#include <algorithm>
#include <functional>
#include "Backend/util_decl/Tensor_container_decl.hpp"

namespace cubby_dnn
{
/// definitions

template <typename T>
Tensor<T>::Tensor(Tensor_type type, const std::vector<int> &shape, int from,
                  bool _mutable, const std::string &name)
    : type(type), from(from), _mutable(_mutable)
{
    this->shape = shape;
    this->name = name;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &rhs)
{
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &&rhs) noexcept
{
}

template <typename T>
struct Tensor_object<T>::storage
{
 private:
    storage(const std::vector<T> &data, const std::vector<int> &shape);

    storage(std::vector<T> &&data, std::vector<int> &&shape);

    std::vector<T> data;     // stores actual data with data type 'T'
    std::vector<int> shape;  // shape of the data
    typedef std::size_t size_type;
    size_type byte_size;
};

template <typename T>
Tensor_object<T>::storage::storage(const std::vector<T> &data,
                                   const std::vector<int> &shape)
{
    this->data = data;
    this->shape = shape;
    byte_size = this->data.size();
}

template <typename T>
Tensor_object<T>::storage::storage(std::vector<T> &&data,
                                   std::vector<int> &&shape)
{
    this->data = std::forward<std::vector<T>>(data);
    this->shape = std::forward<std::vector<int>>(data);
    byte_size = this->data.size();
}

template <typename T>
Tensor_object<T>::Tensor_object(const std::vector<T> &data,
                                const std::vector<int> &shape, Tensor_type type,
                                const std::string &name, int from, int to)
    : type(type), from(from), to(to)
{
    verify<T>(data, shape);  // checks exception if arguments are invalid

    this->tensor_object = std::make_unique<storage>(data, shape);
    this->name = name;
}

template <typename T>
Tensor_object<T>::Tensor_object(std::vector<T> &&data, std::vector<int> &&shape,
                                Tensor_type type, std::string &&name, int from,
                                int to)
    : type(type), from(from), to(to)
{
    verify<T>(data, shape);  // checks exception if arguments are invalid

    this->tensor_object =
        std::make_unique<storage>(std::forward<std::vector<T>>(data),
                                  std::forward<std::vector<int>>(shape));
    this->name = std::forward<std::string>(name);
}

template <typename T>
Tensor_object<T>::Tensor_object(const Tensor_object<T> &rhs)
    : tensor_object(nullptr)
{
    if (!rhs.tensor_object)
        this->tensor_object = std::make_unique<Tensor_object<T>::tensor_object>(
            *rhs.tensor_object);
}

template <typename T>
Tensor_object<T>::Tensor_object(Tensor_object<T> &&rhs) noexcept = default;

template <typename T>
Tensor_object<T> &Tensor_object<T>::operator=(
    const cubby_dnn::Tensor_object<T> &rhs)
{
    // may throw std::bad_alloc() (this function will provide strong guarantee)
    if (rhs.object)
        this->tensor_object = std::make_unique<Tensor_object<T>::tensor_object>(
            *rhs.tensor_object);
    return *this;
}

template <typename T>
Tensor_object<T> &Tensor_object<T>::operator=(
    cubby_dnn::Tensor_object<T> &&rhs) noexcept = default;

template <typename T>
Tensor_object<T>::~Tensor_object() = default;

template <typename T>
void verify(std::vector<T> &data, std::vector<int> &shape)
{
    if (data.empty() || shape.empty())
        std::cout << "empty data" << std::endl;

    unsigned long expected_size = 1;
    for (auto elem : shape)
    {
        expected_size *= elem;
    }

    if (expected_size != data.size())
    {
        std::string err_message = "data shape doesn't match";
        err_message += "Expected Size = " + std::to_string(expected_size);
        err_message += "given data size = " + std::to_string(data.size());
        std::cout << err_message << std::endl;
    }
}

// getters

template <typename T>
long Tensor_object<T>::get_data_size() const
{
    if (!tensor_object)
        std::cout << "tensor_object is empty" << std::endl;
    return static_cast<int>(tensor_object->data.size());
}

template <typename T>
long Tensor_object<T>::get_data_byte_size() const
{
    if (!tensor_object)
        std::cout << "tensor_object is empty" << std::endl;
    return static_cast<long>(tensor_object->data.size() * sizeof(T));
}

template <typename T>
const std::vector<int> &Tensor_object<T>::get_shape() const
{
    if (!tensor_object)
        std::cout << "tensor_object is empty" << std::endl;
    return tensor_object->data.shape();
}

template <typename T>
const std::vector<int> &Tensor_object<T>::get_data() const
{
    if (!tensor_object)
        std::cout << "tensor_object is empty" << std::endl;
    return tensor_object->data;
}

/// management

// TODO: make these thread-safe
template <typename T>
unsigned long Adj_management<T>::add_op_adj()
{
    auto graph_size = adj_forward.size();

    auto expected_row_size = (graph_size + 1 > default_graph_size)
                                 ? graph_size + 1
                                 : default_graph_size;

    std::deque<std::shared_ptr<Tensor_object<T>>> temp(expected_row_size,
                                                       nullptr);

    auto emplace_until_size =
        [expected_row_size, graph_size](
            std::deque<std::shared_ptr<Tensor_object<T>>> &arg) mutable {
            while (expected_row_size > graph_size)
            {
                arg.emplace_back(std::make_shared<Tensor_object<T>>(
                    nullptr));  // copy-construct new temp
                graph_size = arg.size();
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

    return adj_forward.size();
}

template <typename T>
void Adj_management<T>::add_edge(int from, int to,
                                 std::shared_ptr<Tensor_object<T>> &tensor_object_ptr)
{
    auto graph_size = (adj_forward.size());
    if (from == to)
    {
        std::string error_msg = "cannot connect to operation itself";
        std::cout << error_msg << std::endl;
    }

    if (graph_size + 1 < from or graph_size + 1 < to)
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
std::shared_ptr<Tensor_object<T>> Adj_management<T>::get_tensor_ptr(int from,
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
