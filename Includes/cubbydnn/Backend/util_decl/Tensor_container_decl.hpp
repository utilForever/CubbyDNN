//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include "shape.hpp"

namespace cubby_dnn
{
enum class Tensor_type
{
    weight,
    bias,
    filter,
    placeHolder,
    normal,
    None
};

const static long error_id = -1;


template <typename T>
void verify(
    const std::vector<T> &data,
    const std::vector<int> &shape);  // throws exception if input in invalid

template <typename T>
class Tensor_object
{
 public:
    Tensor_object(const std::vector<T> &data, const std::vector<int> &shape,
                  Tensor_type type, int from,
                  int to);  //(1)

    Tensor_object(std::vector<T> &&data, std::vector<int> &&shape,
                  Tensor_type type, int from,
                  int to);  //(2)

    Tensor_object(const Tensor_object<T> &rhs);  //(3)

    Tensor_object(Tensor_object<T> &&rhs) noexcept;  //(4)

    Tensor_object &operator=(const Tensor_object<T> &rhs);  //(5)

    Tensor_object &operator=(Tensor_object<T> &&rhs) noexcept;  //(6)

    ~Tensor_object();

    /*
      for (4), (6) no exceptions can be thrown
      for (3), (5) std::bad_alloc may be thrown
      for (1), (2) if given _data_ or _shape_ is empty,
      _cubby_dnn::ArgumentException_ is thrown if given _shape_ does not match
      size of the data, _cubby_dnn::ArgumentException_ is thrown
     */

 private:

    bool _mutable = true;

    Tensor_type type;

    int from, to;

    struct storage;

    std::unique_ptr<storage> tensor_object;

 public:
    /// getters
    bool has_data() const
    {
        if (!tensor_object)
            return false;
        else
            return true;
    }

    Tensor_type get_type() const
    {
        return type;
    }


    long get_data_size() const;

    long get_data_byte_size() const;

    const std::vector<T> &get_data() const;

    bool is_mutable() const
    {
        return _mutable;
    }

    /// setters
    void disable_training()
    {
        _mutable = false;
    }

    void enable_training()
    {
        _mutable = true;
    }


    void set_type(Tensor_type type)
    {
        this->type = type;
    }

    void make_mutable()
    {
        this->_mutable = true;
    }

    void make_constant()
    {
        this->_mutable = false;
    }

};

template <typename T>
class Tensor
{
 public:
    Tensor(Tensor_type type, const std::vector<int> &shape, long from,
           bool _mutable = true,
           const std::string &name = "Tensor");  //(1)

    Tensor(Tensor<T> &rhs);

    Tensor(Tensor<T> &&rhs) noexcept;

 public:
    /// getters
    bool has_tensor_object()
    {
        return tensor_object_ptr.lock();
    }

    bool is_valid() const
    {
        return !shape.empty();
    }

    Tensor_type get_type() const
    {
        return type;
    }

    const std::string &get_name() const
    {
        return this->name;
    }

    const std::vector<int> &get_shape() const
    {
        return this->shape;
    }

    unsigned long get_data_size()
    {
        return shape::get_shape_size(shape);
    }

    bool is_mutable() const
    {
        return _mutable;
    }

    const std::weak_ptr<Tensor_object<T>> &get_tensor_container_ptr() const
    {
        return tensor_object_ptr;
    }

    long get_from() const
    {
        return from;
    }

    /// setters

    void set_type(Tensor_type type)
    {
        this->type = type;
        if (auto temp_ptr = tensor_object_ptr.lock())
            temp_ptr->set_type(type);
    }

    void set_name(const std::string &name)
    {
        if (auto temp_ptr = tensor_object_ptr.lock())
            temp_ptr->set_name(name);
    }

    void make_mutable()
    {
        this->_mutable = true;
        if (auto temp_ptr = tensor_object_ptr.lock())
            temp_ptr->make_mutable();
    }

    void make_constant()
    {
        this->_mutable = false;
        if (auto temp_ptr = tensor_object_ptr.lock())
            temp_ptr->make_constant();
    }

    // adds operation ids that this tensor heads to
    void add_to(long to)
    {
        this->to_vect.emplace_back(to);
    }

    // adds tensor objects this tensor is giving output of
    void add_tensor_object(const std::shared_ptr<Tensor_object<T>> tensor_ptr)
    {
        this->tensor_object_ptr_vect.emplace_back(tensor_ptr);
    }

 private:
    long from;  // ID of operation that this tensor is generated

    std::vector<long> to_vect; // vector for storing operations this tensor will head to

    std::vector<int> shape; // shape of this tensor

    bool _mutable =
        true;  // determines whether data of this tensor can be modified
    // properties of the tensor
    std::string name;

    Tensor_type type =
        Tensor_type::None;  // type of the tensor_container it is pointing to

    std::weak_ptr<Tensor_object<T>>
        tensor_object_ptr;  // weak pointer pointing to tensor object

    std::vector<std::weak_ptr<Tensor_object<T>>> tensor_object_ptr_vect;
};

/// Resource management

template <typename T>
class Adj_management
{
 public:
    /// Adds new operation
    static unsigned long add_op_adj();
    /// Adds new edge between two
    static void add_edge(long from, long to,
                         std::shared_ptr<Tensor_object<T>> &tensor_object_ptr);

    static unsigned long get_graph_size()
    {
        return adj_forward.size();
    }

    static std::shared_ptr<Tensor_object<T>> get_tensor_ptr(int from, int to);

 private:
    static constexpr int default_graph_size = 30;

    static std::deque<std::deque<std::shared_ptr<Tensor_object<T>>>>
        adj_forward;

    Adj_management() = default;  /// disable the constructor

    static std::mutex adj_mutex;  // mutex for restricting access to adj matrix
};

template <typename T>
std::deque<std::deque<std::shared_ptr<Tensor_object<T>>>>
    Adj_management<T>::adj_forward;

template <typename T>
std::mutex Adj_management<T>::adj_mutex;
}  // namespace cubby_dnn

#endif  // CUBBYDNN_BACKEND_H
