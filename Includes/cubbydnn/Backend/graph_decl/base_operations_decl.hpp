
#ifndef CUBBYDNN_BASE_OPERATIONS_DECL_HPP
#define CUBBYDNN_BASE_OPERATIONS_DECL_HPP

#include "Backend/graph/tensor.hpp"
#include "Backend/util/stream.hpp"

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

    void add_output(std::shared_ptr<tensor_object<T>> tensor_ptr)
    {
        output_tensor_vector.emplace_back(tensor_ptr);
    }

    void add_input(std::shared_ptr<tensor_object<T>> tensor_ptr)
    {
        input_tensor_vector.emplace_back(tensor_ptr);
    }

    size_t input_vector_size()
    {
        return input_tensor_vector.size();
    }

    size_t output_vector_size()
    {
        return output_tensor_vector.size();
    }

    const std::string &get_name()
    {
        return name;
    }

    long get_id()
    {
        return operation_id;
    }

    const std::string print_info()
    {
        std::string info = name;
        info += "\noperation id: " + std::to_string(operation_id);
        info +=
            "\ninput tensor num: " + std::to_string(input_tensor_vector.size());
        info +=
            "\noutput tensor num: " + std::to_string(output_tensor_vector.size());
        return info;
    }

    operation_info get_info() const
    {
        return operation_info(operation_id, input_tensor_vector.size(),
                              output_tensor_vector.size(), name);
    }

    decltype(auto) get_input_tensor_vector() const
    {
        return input_tensor_vector;
    }

    decltype(auto) get_output_tensor_vector() const
    {
        return output_tensor_vector;
    }

 protected:
    explicit operation();

 protected:
    operation_type op_type;
    long operation_id = 0;
    std::vector<std::shared_ptr<tensor_object<T>>> input_tensor_vector;
    std::vector<std::shared_ptr<tensor_object<T>>> output_tensor_vector;
    std::string name;
};

template <typename T>
class empty_op : public operation<T>
{
 public:
    explicit empty_op()
    {
        this->op_type = operation_type::empty;
        this->name = "Empty operation";
    }
};

template <typename T>
class mat_mul_op : public operation<T>
{
 public:

    explicit mat_mul_op(long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }

};

template <typename T>
class mat_add_op : public operation<T>
{
 public:

    explicit mat_add_op(long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }

};

template <typename T>
class mat_dot_op : public operation<T>
{
 public:
    explicit mat_dot_op(long operation_id, const std::string &name, T multiplier)
    {
        this->operation_id = operation_id;
        this->name = name;
        this->multiplier = multiplier;
    }
private:
    T multiplier;
};

template <typename T>
class reshape_op : public operation<T>
{
 public:
    explicit reshape_op(long operation_id, const std::string &name, const tensor_shape &shape)
    {
        this->operation_id = operation_id;
        this->name = name;
        this->shape = shape;
    }

 private:
    tensor_shape shape;
};

template <typename T>
class placeholder_op : public operation<T>
{
 public:
    explicit placeholder_op(long operation_id, const tensor_shape &shape, stream<T> &stream, const std::string &name)
    {
        this->operation_id = operation_id;
        this->data_stream = stream;
        this->name = name;
        this->shape = shape;
    }

 private:
    tensor_shape shape;
    stream<T> data_stream;
};

template <typename T>
class weight_op : public operation<T>
{
 public:
    explicit weight_op(long operation_id, const tensor_shape &shape, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
        this->shape = shape;
    }

 private:
    tensor_shape shape;
};

template <typename T>
class constant_op : public operation<T>
{
 public:
    explicit constant_op(long operation_id, const tensor_shape &shape, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
        this->shape = shape;
    }

 private:
    tensor_shape shape;
};

template <typename T>
class wrapper_op : public operation<T>
{
 public:
    explicit wrapper_op(long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }
};

template <typename T>
class operation_management
{
 public:
    static void add_op(operation<T> operation);
    static void set_op(unsigned int id, const operation<T> &operation);
    static void add_output_of(long id,
                              std::shared_ptr<tensor_object<T>> tensor_ptr);
    static void add_input_of(long id,
                             std::shared_ptr<tensor_object<T>> tensor_ptr);

    static void print_operation_info()
    {
        for (auto op : operation_list)
        {
            std::cout << op.print_info() << std::endl;
        }
    }

    static const auto get_operation_info()
    {
        std::vector<operation_info> op_vector;
        for (operation<T> operation : operation_list)
        {
            op_vector.emplace_back(operation.get_info());
        }
        return op_vector;
    }

    static long number_of_operations()
    {
        return static_cast<long>(operation_list.size());
    }

    static void create_adj()
    {
        adj_management<T>::reserve_adj(
            static_cast<long>(operation_list.size()));
        for (operation<T> operation : operation_list)
        {
            decltype(auto) input_tensor_vect =
                    operation.get_input_tensor_vector();
            for (auto tensor_ptr : input_tensor_vect)
            {
                adj_management<T>::add_edge(tensor_ptr->get_from(),
                                            tensor_ptr->get_to(), tensor_ptr);
            }
        }
    }

 private:
    static operation<T> &get_op(long operation_id)
    {
        for (decltype(auto) operation : operation_list)
        {
            if (operation.get_id() == operation_id)
                return operation;
        }
        // returns empty operation if nothing is found
        return operation_list[0];
    }
    static std::deque<operation<T>> operation_list;
    static std::mutex operation_list_mutex;
};

template <typename T>
std::deque<operation<T>> operation_management<T>::operation_list =
    std::deque<operation<T>>();

template <typename T>
std::mutex operation_management<T>::operation_list_mutex = std::mutex();

}  // namespace cubby_dnn

#endif