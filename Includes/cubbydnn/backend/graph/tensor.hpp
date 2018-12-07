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
template <typename T>
struct tensor_data<T>::data
{
 public:
    data(const std::vector<T> &data, const tensor_shape &shape);

    data(std::vector<T> &&data, tensor_shape &&shape);

    std::vector<T> data_vector;
    tensor_shape shape;
    size_t byte_size{};
};

template <typename T>
tensor_data<T>::data::data(const std::vector<T> &data,
                                   const tensor_shape &shape)
{
    this->data_vector = data;
    this->shape = shape;
    byte_size = this->data_vector.size();
}

template <typename T>
tensor_data<T>::data::data(std::vector<T> &&data, tensor_shape &&shape)
{
    this->data_vector = std::forward<std::vector<T>>(data);
    this->shape = std::forward<tensor_shape>(data);
    byte_size = this->data_vector.size();
}


template <typename T>
tensor_data<T>::tensor_data(size_t data_size, const tensor_shape &shape,
                                long from, long to)
    : from(from), to(to)
{
    std::vector<T> data_vector(data_size);
    verify<T>(data_vector, shape);

    this->tensor_storage = std::make_unique<data>(data_vector, shape);
}

template <typename T>
tensor_data<T>::tensor_data(size_t data_size, tensor_shape &&shape,
                                long from, long to)
    : from(from), to(to)
{
    std::vector<T> data(data_size);
    verify<T>(data, shape);

    this->tensor_storage = std::make_unique<data>(
        std::forward<std::vector<T>>(data), std::forward<tensor_shape>(shape));
}

template <typename T>
tensor_data<T>::tensor_data(const tensor_data<T> &rhs)
{
    if (!rhs.tensor_storage)
        this->tensor_storage =
            std::make_unique<tensor_data<T>::data>(rhs.get_data(), rhs.get_data_shape());
    _mutable = rhs._mutable;
    from = rhs.from;
    to = rhs.to;
}

template <typename T>
tensor_data<T>::tensor_data(tensor_data<T> &&rhs) noexcept
{
    if (rhs.tensor_storage)
        this->tensor_storage = std::move(rhs.tensor_storage);
    this->_mutable = rhs._mutable;
    this->from = rhs.from;
    this->to = rhs.to;
}

template <typename T>
tensor_data<T> &tensor_data<T>::operator=(
    const cubby_dnn::tensor_data<T> &rhs)
{
    /// may throw std::bad_alloc() (this function will provide strong guarantee)
    if (rhs.object)
        this->tensor_storage =
            std::make_unique<tensor_data<T>::tensor_storage>(
                *rhs.tensor_storage);
    return *this;
}

template <typename T>
tensor_data<T> &tensor_data<T>::operator=(
    cubby_dnn::tensor_data<T> &&rhs) noexcept = default;

template <typename T>
tensor_data<T>::~tensor_data() = default;

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
bool tensor_data<T>::has_data() const
{
    if (!tensor_storage)
        return false;
    else
        return true;
}

template <typename T>
size_t tensor_data<T>::get_data_size() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_data is empty" << std::endl;
        return error_id;
    }
    return tensor_storage->data_vector.size();
}

template <typename T>
size_t tensor_data<T>::get_data_byte_size() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_data is empty" << std::endl;
        return error_id;
    }
    return tensor_storage->data_vector.size() * sizeof(T);
}

template <typename T>
const std::vector<T> tensor_data<T>::get_data() const
{
    if (!tensor_storage)
    {
        std::cout << "tensor_data is empty" << std::endl;
        return std::vector<T>();
    }
    return tensor_storage->data_vector;
}

template<typename T>
tensor_shape tensor_data<T>::get_data_shape() const{
    if (!tensor_storage)
    {
        std::cout << "tensor_data is empty" << std::endl;
        return tensor_shape();
    }
    return tensor_storage->shape;
}

template <typename T>
bool tensor_data<T>::is_mutable() const
{
    return _mutable;
}

template <typename T>
void tensor_data<T>::make_mutable()
{
    this->_mutable = true;
}

template <typename T>
void tensor_data<T>::make_constant()
{
    this->_mutable = false;
}

template <typename T>
long tensor_data<T>::comes_from() const
{
    return from;
}

template <typename T>
long tensor_data<T>::heads_to() const
{
    return to;
}

template <typename T>
tensor<T>::tensor(const tensor_shape &shape, long from, bool _mutable)
    : from(from), _mutable(_mutable), shape(shape)
{
}

template <typename T>
bool tensor<T>::is_valid() const
{
    return !shape.empty();
};

template <typename T>
const tensor_shape &tensor<T>::get_shape() const
{
    return this->shape;
}

template <typename T>
size_t tensor<T>::get_data_size() const
{
    return shape.size();
}

template <typename T>
bool tensor<T>::is_mutable() const
{
    return _mutable;
}

template <typename T>
long tensor<T>::get_from() const
{
    return from;
}

template <typename T>
void tensor<T>::make_mutable()
{
    this->_mutable = true;
}

template <typename T>
void tensor<T>::make_constant()
{
    this->_mutable = false;
}

template <typename T>
void tensor<T>::add_to(long to)
{
    this->to_vector.emplace_back(to);
}

}  // namespace cubby_dnn
#endif  // CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
