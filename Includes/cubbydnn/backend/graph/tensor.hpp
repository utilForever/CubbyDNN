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

    storage(std::vector<T> &&data, tensor_shape &&shape);

    std::vector<T> data;  // stores actual data with data type 'T'
    tensor_shape shape;   // shape of the data
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
tensor_object<T>::storage::storage(std::vector<T> &&data, tensor_shape &&shape)
{
    this->data = std::forward<std::vector<T>>(data);
    this->shape = std::forward<tensor_shape>(data);
    byte_size = this->data.size();
}

template <typename T>
tensor_object<T>::tensor_object(size_t data_size, const tensor_shape &shape,
                                tensor_type type, long from, long to)
    : type(type), from(from), to(to)
{
    std::vector<T> data(data_size);
    verify<T>(data, shape);

    this->tensor_storage = std::make_unique<storage>(data, shape);
}

template <typename T>
tensor_object<T>::tensor_object(size_t data_size, tensor_shape &&shape,
                                tensor_type type, long from, long to)
    : type(type), from(from), to(to)
{
    std::vector<T> data(data_size);
    verify<T>(data, shape);  // checks exception if arguments are invalid

    this->tensor_storage = std::make_unique<storage>(
        std::forward<std::vector<T>>(data), std::forward<tensor_shape>(shape));
}

template <typename T>
tensor_object<T>::tensor_object(const tensor_object<T> &rhs) {
    if(!rhs.tensor_storage)
        this->tensor_storage = std::make_unique<tensor_object<T>::storage>(*rhs.tensor_storage);
    _mutable = rhs._mutable;
    type = rhs.type;
    from = rhs.from;
    to = rhs.to;
}

template <typename T>
tensor_object<T>::tensor_object(tensor_object<T> &&rhs) noexcept
{
    if (!rhs.tensor_storage)
        this->tensor_storage = std::move(rhs.tensor_storage);
}

template <typename T>
tensor_object<T> &tensor_object<T>::operator=(
    const cubby_dnn::tensor_object<T> &rhs)
{
    // may throw std::bad_alloc() (this function will provide strong guarantee)
    if (rhs.object)
        this->tensor_storage =
            std::make_unique<tensor_object<T>::tensor_storage>(
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
}  // namespace cubby_dnn
#endif  // CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
