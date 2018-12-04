
#ifndef CUBBYDNN_BASE_OPERATIONS_DECL_HPP
#define CUBBYDNN_BASE_OPERATIONS_DECL_HPP

#include "backend/graph/tensor.hpp"
#include "backend/util/stream.hpp"

namespace cubby_dnn
{
enum class operation_type
{
    start,
    middle,
    final,
    empty
};

struct operation_info
{
 public:
    operation_info(long operation_id, size_t input_size,
                   size_t output_size, const std::string &name)
        : operation_id(operation_id),
          input_size(input_size),
          output_size(output_size),
          name(name)
    {
    }

    bool operator==(const operation_info &rhs) const
    {
        return operation_id == rhs.operation_id &&
               input_size == rhs.input_size && output_size == rhs.output_size &&
               name == rhs.name;
    }

    bool operator!=(const operation_info &rhs) const
    {
        return !(rhs == *this);
    }

    long operation_id;
    size_t input_size;
    size_t output_size;
    std::string name;
};

template <typename T>
class operation
{
 public:

    void add_output(long tensor_id);
    void add_input(long tensor_id);

    size_t input_vector_size();

    size_t output_vector_size();

    const std::string &get_name();

    long get_id();

    const std::string print_info();

    operation_info get_info() const;

    decltype(auto) get_input_tensor_vector() const;
    decltype(auto) get_output_tensor_vector() const;

 protected:
    explicit operation();
    operation_type op_type = operation_type::empty;
    long operation_id = 0;
    std::vector<long> input_tensor_id_vector;
    std::vector<long> output_tensor_id_vector;
    std::string name;
};

template <typename T>
class empty_op : public operation<T>
{
 public:
    explicit empty_op();
};

template <typename T>
class mat_mul_op : public operation<T>
{
 public:

    explicit mat_mul_op(long operation_id, const std::string &name);

};

template <typename T>
class mat_add_op : public operation<T>
{
 public:

    explicit mat_add_op(long operation_id, const std::string &name);

};

template <typename T>
class mat_dot_op : public operation<T>
{
 public:
    explicit mat_dot_op(long operation_id, const std::string &name, T multiplier);
private:
    T multiplier;
};

template <typename T>
class reshape_op : public operation<T>
{
 public:
    explicit reshape_op(long operation_id, const std::string &name, const tensor_shape &shape);

 private:
    tensor_shape shape;
};

template <typename T>
class placeholder_op : public operation<T>
{
 public:
    explicit placeholder_op(long operation_id, const tensor_shape &shape, stream<T> &stream, const std::string &name);

 private:
    tensor_shape shape;
    stream<T> data_stream;
};

template <typename T>
class weight_op : public operation<T>
{
 public:
    explicit weight_op(long operation_id, const tensor_shape &shape, const std::string &name);

 private:
    tensor_shape shape;
};

template <typename T>
class constant_op : public operation<T>
{
 public:
    explicit constant_op(long operation_id, const tensor_shape &shape, const std::string &name);

 private:
    tensor_shape shape;
};

template <typename T>
class wrapper_op : public operation<T>
{
 public:
    explicit wrapper_op(long operation_id, const std::string &name);
};

}  // namespace cubby_dnn

#endif