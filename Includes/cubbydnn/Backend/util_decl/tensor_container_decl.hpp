//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include "shape_checker.hpp"

namespace cubby_dnn
{
enum class tensor_type
{
    variable,
    placeHolder,
    normal,
    None
};

const static long error_id = -1;

template <typename T>
bool verify(const std::vector<T> &data, const tensor_shape &shape);

template <typename T>
class tensor_object
{
 public:
    tensor_object(const std::vector<T> &data, const tensor_shape &shape,
                  tensor_type type, long from,
                  long to);  //(1)

    tensor_object(std::vector<T> &&data, tensor_shape &&shape, tensor_type type,
                  long from,
                  long to);  //(2)

    tensor_object(const tensor_object<T> &rhs);  //(3)

    tensor_object(tensor_object<T> &&rhs) noexcept;  //(4)

    tensor_object &operator=(const tensor_object<T> &rhs);  //(5)

    tensor_object &operator=(tensor_object<T> &&rhs) noexcept;  //(6)

    ~tensor_object();

    /*
      for (4), (6) no exceptions can be thrown
      for (3), (5) std::bad_alloc may be thrown
      for (1), (2) if given _data_ or _shape_ is empty,
      _cubby_dnn::ArgumentException_ is thrown if given _shape_ does not match
      size of the data, _cubby_dnn::ArgumentException_ is thrown
     */

 private:
    bool _mutable = true;

    tensor_type type;

    long from, to;

    struct storage;

    std::unique_ptr<storage> tensor_storage;

 public:
    /// getters
    bool has_data() const
    {
        if (!tensor_storage)
            return false;
        else
            return true;
    }

    tensor_type get_type() const
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

    void set_type(tensor_type type)
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

    long get_from() const
    {
        return from;
    };

    long get_to() const
    {
        return to;
    };
};

template <typename T>
class tensor
{
 public:
    tensor(tensor_type type, const tensor_shape &shape, long from,
           bool _mutable = true,
           const std::string &name = "Tensor");  //(1)

    tensor(tensor<T> &rhs);

    tensor(tensor<T> &&rhs) noexcept;

 public:
    ///getters

    bool is_valid() const
    {
        return !shape.empty();
    }

    tensor_type get_type() const
    {
        return type;
    }

    const std::string &get_name() const
    {
        return this->name;
    }

    const tensor_shape &get_shape() const
    {
        return this->shape;
    }

    long get_data_size() const
    {
        return shape.size();
    }

    bool is_mutable() const
    {
        return _mutable;
    }


    long get_from() const
    {
        return from;
    }

    /// setters

    void set_name(const std::string &name){
        this->name = name;
    }

    void set_type(tensor_type type)
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

    // adds operation ids that this tensor heads to
    void add_to(long to)
    {
        this->to_vect.emplace_back(to);
    }


 private:
    long from;  // ID of operation that this tensor is generated

    std::vector<long>
        to_vect;  // vector for storing operations this tensor will head to

    tensor_shape shape;  // shape of this tensor

    bool _mutable =
        true;  // determines whether data of this tensor can be modified
    // properties of the tensor
    std::string name;

    tensor_type type =
        tensor_type::None;  // type of the tensor_container it is pointing to

    // weak pointer pointing to tensor object
};

/// Resource management

template <typename T>
class adj_management
{
 public:
    /// Adds new operation
    static unsigned long add_op_adj();
    /// Adds new edge between two
    static void add_edge(long from, long to,
                         std::shared_ptr<tensor_object<T>> &tensor_object_ptr);

    static unsigned long get_graph_size()
    {
        return adj_forward.size();
    }

    static std::shared_ptr<tensor_object<T>> get_tensor_ptr(int from, int to);

    static void print_adj()
    {
        std::cout << "--Adjacency Matrix--" << std::endl;
        for (auto row : adj_forward)
        {
            for (auto col : row)
            {
                if (col)
                    std::cout << col->get_from() << " ";
                else
                    std::cout << "*"
                              << " ";
            }
            std::cout << std::endl;
        }
    }

    static void reserve_adj(long size)
    {
        for (long i = 0; i < size; i++)
        {
            add_op_adj();  // increment size of adj matrix
        }
    }

 private:
    static constexpr int default_graph_size = 0;

    static std::deque<std::deque<std::shared_ptr<tensor_object<T>>>>
        adj_forward;

    adj_management() = default;  /// disable the constructor

    static std::mutex adj_mutex;  // mutex for restricting access to adj matrix
};

template <typename T>
std::deque<std::deque<std::shared_ptr<tensor_object<T>>>>
    adj_management<T>::adj_forward;

template <typename T>
std::mutex adj_management<T>::adj_mutex;
}  // namespace cubby_dnn

#endif  // CUBBYDNN_BACKEND_H
