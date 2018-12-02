
#ifndef CUBBYDNN_BASE_OPERATIONS_DECL_HPP
#define CUBBYDNN_BASE_OPERATIONS_DECL_HPP

#include "Backend/util/Tensor_container.hpp"
#include "Backend/util_decl/stream_decl.hpp"

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
    operation_info(unsigned long operation_id, size_t input_size,
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

    unsigned long operation_id;
    size_t input_size;
    size_t output_size;
    std::string name;
};

template <typename T>
class operation
{
 public:
    void set_input_vect(
        std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect)
    {
        input_tensor_vect = tensor_vect;
    }

    void set_output_vect(
        std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect)
    {
        output_tensor_vect = tensor_vect;
    }

    void add_output(std::shared_ptr<tensor_object<T>> tensor_ptr)
    {
        output_tensor_vect.emplace_back(tensor_ptr);
    }

    void add_input(std::shared_ptr<tensor_object<T>> tensor_ptr)
    {
        input_tensor_vect.emplace_back(tensor_ptr);
    }

    bool has_input_vector()
    {
        return !input_tensor_vect.empty();
    }

    bool has_output_vector()
    {
        return !output_tensor_vect.empty();
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
            "\ninput tensor num: " + std::to_string(input_tensor_vect.size());
        info +=
            "\noutput tensor num: " + std::to_string(output_tensor_vect.size());
        return info;
    }

    operation_info get_info() const
    {
        return operation_info(operation_id, input_tensor_vect.size(),
                              output_tensor_vect.size(), name);
    }

    decltype(auto) get_input_tensor_vect() const
    {
        return input_tensor_vect;
    }

 protected:
    explicit operation();

 protected:
    operation_type op_type;
    unsigned long operation_id = 0;
    std::vector<std::shared_ptr<tensor_object<T>>> input_tensor_vect;
    std::vector<std::shared_ptr<tensor_object<T>>> output_tensor_vect;
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
    explicit mat_mul_op(std::shared_ptr<tensor_object<T>> tensor1,
                        std::shared_ptr<tensor_object<T>> tensor2,
                        std::shared_ptr<tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matmul");

    explicit mat_mul_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

    explicit mat_mul_op(
        std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect,
        std::shared_ptr<tensor_object<T>> output_tensor,
        unsigned long operation_id, const std::string &name = "Matmul");
};

template <typename T>
class mat_add_op : public operation<T>
{
 public:
    explicit mat_add_op(std::shared_ptr<tensor_object<T>> tensor1,
                        std::shared_ptr<tensor_object<T>> tensor2,
                        std::shared_ptr<tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matadd");

    explicit mat_add_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

    explicit mat_add_op(
        std::vector<std::shared_ptr<tensor_object<T>>> tensor_vect,
        std::shared_ptr<tensor_object<T>> output_tensor,
        unsigned long operation_id, const std::string &name = "Matadd");
};

template <typename T>
class mat_dot_op : public operation<T>
{
 public:
    explicit mat_dot_op(std::shared_ptr<tensor_object<T>> tensor1,
                        std::shared_ptr<tensor_object<T>> identity_tensor,
                        std::shared_ptr<tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matdot");

    explicit mat_dot_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation
};

template <typename T>
class reshape_op : public operation<T>
{
 public:
    explicit reshape_op(std::shared_ptr<tensor_object<T>> tensor1,
                        std::shared_ptr<tensor_object<T>> output_tensor,
                        const tensor_shape &shape, unsigned long operation_id,
                        const std::string &name = "reshape");

    explicit reshape_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    tensor_shape shape;
};

template <typename T>
class placeholder_op : public operation<T>
{
 public:
    explicit placeholder_op(std::shared_ptr<tensor_object<T>> output_tensor,
                            const tensor_shape &shape, unsigned long operation_id,
                            const std::string &name = "placeHolder");

    explicit placeholder_op(unsigned long operation_id, stream<T> &stream,
                            const std::string &name)
    {
        this->operation_id = operation_id;
        this->data_stream = stream;
        this->name = name;
    }  // empty constructor for operation

 private:
    tensor_shape shape;
    stream<T> data_stream;
};

template <typename T>
class weight_op : public operation<T>
{
 public:
    explicit weight_op(std::shared_ptr<tensor_object<T>> output_tensor,
                       const tensor_shape &shape, unsigned long operation_id,
                       const std::string &name = "weight");

    explicit weight_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    tensor_shape shape;
};

template <typename T>
class constant_op : public operation<T>
{
 public:
    explicit constant_op(std::shared_ptr<tensor_object<T>> output_tensor,
                         const tensor_shape &shape, unsigned long operation_id,
                         const std::string &name = "constant");

    explicit constant_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    tensor_shape shape;
};

template <typename T>
class wrapper_op : public operation<T>
{
 public:
    explicit wrapper_op(std::shared_ptr<tensor_object<T>> input_tensor,
                        unsigned long operation_id,
                        const std::string &name = "constant");
    explicit wrapper_op(unsigned long operation_id, const std::string &name)
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

    static unsigned long number_of_operations()
    {
        return static_cast<unsigned long>(operation_list.size());
    }

    static void create_adj()
    {
        adj_management<T>::reserve_adj(
            static_cast<long>(operation_list.size()));
        for (operation<T> operation : operation_list)
        {
            decltype(auto) input_tensor_vect =
                operation.get_input_tensor_vect();
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