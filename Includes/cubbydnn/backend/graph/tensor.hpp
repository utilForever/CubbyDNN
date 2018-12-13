//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
#define CUBBYDNN_TENSOR_CONTAINER_DEF_HPP

#include <algorithm>
#include <functional>

#include "backend/graph_decl/tensor_decl.hpp"
#include "backend/util/terminal.hpp"

namespace cubby_dnn
{
template <typename T>
struct tensor_object<T>::data
{
 public:
    data(const std::vector<T>& data, const tensor_shape& shape);

    data(std::vector<T>&& data, tensor_shape&& shape);

    std::vector<T> data_vector;
    tensor_shape shape;
    bool _mutable = true;
};

template <typename T>
struct tensor_object<T>::info
{
 public:
    info() = default;

    info(long from, long to, bool _mutable = true) : from(from), to(to)
    {
    }

    long from, to;

    bool busy = false;
    /// this field indicates the state of this data
    /// this tensor_object is only allowed to be executed by
    /// operation with same process_count variable
    unsigned process_count = 0;
};

template <typename T>
tensor_object<T>::data::data(const std::vector<T>& data,
                             const tensor_shape& shape)
{
    this->data_vector = data;
    this->shape = shape;
    this->data_vector.size();
}

template <typename T>
tensor_object<T>::data::data(std::vector<T>&& data, tensor_shape&& shape)
{
    this->data_vector = std::forward<std::vector<T>>(data);
    this->shape = std::forward<tensor_shape>(data);
    this->data_vector.size();
}

template <typename T>
tensor_object<T>::tensor_object(size_t data_size, const tensor_shape& shape,
                                long from, long to)
{
    std::vector<T> data_vector(data_size);
    verify<T>(data_vector, shape);

    this->information = info(from, to);
    this->tensor_data = std::make_unique<data>(data_vector, shape);
}

template <typename T>
tensor_object<T>::tensor_object(size_t data_size, tensor_shape&& shape,
                                long from, long to)
{
    std::vector<T> data(data_size);
    verify<T>(data, shape);

    this->information = info(from, to);
    this->tensor_data = std::make_unique<data>(
        std::forward<std::vector<T>>(data), std::forward<tensor_shape>(shape));
}

template <typename T>
tensor_object<T>::tensor_object(const tensor_object<T>& rhs)
{
    if (!rhs.tensor_data)
        this->tensor_data = std::make_unique<tensor_object<T>::data>(
            rhs.get_data_vector(), rhs.get_data_shape());
    this->information = rhs.information;
}

template <typename T>
tensor_object<T>::tensor_object(tensor_object<T>&& rhs) noexcept
{
    if (rhs.tensor_data)
        this->tensor_data = std::move(rhs.tensor_data);
    this->information = rhs.information;
}

template <typename T>
tensor_object<T>& tensor_object<T>::operator=(
    const cubby_dnn::tensor_object<T>& rhs)
{
    /// may throw std::bad_alloc() (this function will provide strong guarantee)
    if (rhs.object)
        this->tensor_data =
            std::make_unique<tensor_object<T>::tensor_data>(*rhs.tensor_data);
    this->information = rhs.information;
    return *this;
}

template <typename T>
tensor_object<T>& tensor_object<T>::operator=(
    cubby_dnn::tensor_object<T>&& rhs) noexcept = default;

template <typename T>
tensor_object<T>::~tensor_object() = default;

template <typename T>
bool verify(const std::vector<T>& data, const tensor_shape& shape)
{

    if (data.size() != shape.size())
    {
        std::string err_message = "data shape doesn't match";
        err_message += "Expected Size = " + std::to_string(shape.size());
        err_message += "given data size = " + std::to_string(data.size());
        terminal::print_error(err_type::shape_matching, "verify", err_message);
        std::cout << err_message << std::endl;
        return false;
    }
    return true;
}

// getters

template <typename T>
const typename tensor_object<T>::info& tensor_object<T>::get_information() const
{
    return information;
}

template <typename T>
const std::vector<T> tensor_object<T>::get_data_vector() const
{
    if (!tensor_data)
    {
        std::string msg = "trying to access empty tensor_data";
        terminal::print_error(err_type::memory_error, "tensor_object<T>::g_vector", msg);
        return std::vector<T>();
    }
    return tensor_data->data_vector;
}

template <typename T>
std::unique_ptr<typename tensor_object<T>::data>
tensor_object<T>::get_data_ptr()
{
    if (!tensor_data || information.busy)
    {
        std::string msg = "trying to access empty tensor_data";
        terminal::print_error(err_type::memory_error, "tensor_object<T>::get_data_ptr",
                    msg);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(lock_tensor_storage);
    information.busy = true;
    return std::move(tensor_data);
}

template <typename T>
void tensor_object<T>::return_data_ptr(
    std::unique_ptr<typename tensor_object<T>::data> rhs)
{
    if (tensor_data || !information.busy)
    {
        std::string msg = "trying to access empty tensor_data";
        terminal::print_error(err_type::memory_error, "tensor_object<T>::return_data_ptr",
                    msg);
        return;
    }

    std::lock_guard<std::mutex> lock(lock_tensor_storage);
    information.busy = false;
    tensor_data = std::move(rhs);
}

template <typename T>
tensor_shape tensor_object<T>::get_data_shape() const
{
    if (!tensor_data)
    {
        std::string msg = "trying to access empty tensor_data";
        terminal::print_error(err_type::memory_error, "tensor_object<T>::get_data_shape",
                    msg);
        return tensor_shape();
    }
    return tensor_data->shape;
}

template <typename T>
void tensor_object<T>::set_constant()
{
    if (tensor_data && information.busy == false)
        tensor_data->_mutable = false;
    else{
        std::string msg = "fail to set tensor_object constant";
        terminal::print_error(err_type::memory_error, "tensor_object<T>::constant",
                msg);
    }
}

template <typename T>
unsigned tensor_object<T>::get_process_count()
{
    return information.process_count;
}

template <typename T>
void tensor_object<T>::increment_process_count()
{
    information.process_count += 1;
}

template <typename T>
tensor<T>::tensor(const tensor_shape& shape, long from, bool _mutable)
    : from(from), _mutable(_mutable), shape(shape)
{
}

template <typename T>
bool tensor<T>::is_valid() const
{
    return !shape.empty();
};

template <typename T>
const tensor_shape& tensor<T>::get_shape() const
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
