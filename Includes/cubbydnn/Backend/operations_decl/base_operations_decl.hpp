
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

template <typename T>
class Operation
{
 public:
    void set_input_vect(
        std::vector<std::shared_ptr<Tensor_object<T>>> tensor_vect)
    {
        input_tensor_vect = tensor_vect;
    }

    void set_output_vect(
        std::vector<std::shared_ptr<Tensor_object<T>>> tensor_vect)
    {
        output_tensor_vect = tensor_vect;
    }

    void add_output(std::shared_ptr<Tensor_object<T>> tensor_ptr)
    {
        output_tensor_vect.emplace_back(tensor_ptr);
    }

    void add_input(std::shared_ptr<Tensor_object<T>> tensor_ptr)
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
        info+="\noperation id: " + std::to_string(operation_id);
        info +=
            "\ninput tensor num: " + std::to_string(input_tensor_vect.size());
        info +=
            "\noutput tensor num: " + std::to_string(output_tensor_vect.size());
        return info;
    }

    std::tuple<long,unsigned long, unsigned long> get_info() const {
        return std::tuple{operation_id, input_tensor_vect.size(), output_tensor_vect.size()};
    }

    decltype(auto) get_input_tensor_vect() const {
        return input_tensor_vect;
    }

 protected:
    explicit Operation();

 protected:
    operation_type op_type;
    unsigned long operation_id = 0;
    std::vector<std::shared_ptr<Tensor_object<T>>> input_tensor_vect;
    std::vector<std::shared_ptr<Tensor_object<T>>> output_tensor_vect;
    std::string name;
};

template <typename T>
class Empty_op : public Operation<T>
{
 public:
    explicit Empty_op()
    {
        this->op_type = operation_type::empty;
        this->name = "Empty operation";
    }
};

template <typename T>
class Mat_mul_op : public Operation<T>
{
 public:
    explicit Mat_mul_op(std::shared_ptr<Tensor_object<T>> tensor1,
                        std::shared_ptr<Tensor_object<T>> tensor2,
                        std::shared_ptr<Tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matmul");

    explicit Mat_mul_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

    explicit Mat_mul_op(
        std::vector<std::shared_ptr<Tensor_object<T>>> tensor_vect,
        std::shared_ptr<Tensor_object<T>> output_tensor,
        unsigned long operation_id, const std::string &name = "Matmul");
};

template <typename T>
class Mat_add_op : public Operation<T>
{
 public:
    explicit Mat_add_op(std::shared_ptr<Tensor_object<T>> tensor1,
                        std::shared_ptr<Tensor_object<T>> tensor2,
                        std::shared_ptr<Tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matadd");

    explicit Mat_add_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

    explicit Mat_add_op(
        std::vector<std::shared_ptr<Tensor_object<T>>> tensor_vect,
        std::shared_ptr<Tensor_object<T>> output_tensor,
        unsigned long operation_id, const std::string &name = "Matadd");
};

template <typename T>
class Mat_dot_op : public Operation<T>
{
 public:
    explicit Mat_dot_op(std::shared_ptr<Tensor_object<T>> tensor1,
                        std::shared_ptr<Tensor_object<T>> identity_tensor,
                        std::shared_ptr<Tensor_object<T>> output_tensor,
                        unsigned long operation_id,
                        const std::string &name = "Matdot");

    explicit Mat_dot_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation
};

template <typename T>
class Reshape_op : public Operation<T>
{
 public:
    explicit Reshape_op(std::shared_ptr<Tensor_object<T>> tensor1,
                        std::shared_ptr<Tensor_object<T>> output_tensor,
                        const Shape &shape,
                        unsigned long operation_id,
                        const std::string &name = "reshape");

    explicit Reshape_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    Shape shape;
};

template <typename T>
class placeHolder_op : public Operation<T>
{
 public:
    explicit placeHolder_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                            const Shape &shape,
                            unsigned long operation_id,
                            const std::string &name = "placeHolder");

    explicit placeHolder_op(unsigned long operation_id, Stream<T> &stream,
                            const std::string &name)
    {
        this->operation_id = operation_id;
        this->stream = stream;
        this->name = name;
    }  // empty constructor for operation

 private:
    Shape shape;
    Stream<T> stream;
};

template <typename T>
class weight_op : public Operation<T>
{
 public:
    explicit weight_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                       const Shape &shape,
                       unsigned long operation_id,
                       const std::string &name = "weight");

    explicit weight_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    Shape shape;
};

template <typename T>
class constant_op : public Operation<T>
{
 public:
    explicit constant_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                         const Shape &shape,
                         unsigned long operation_id,
                         const std::string &name = "constant");

    explicit constant_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    Shape shape;
};

template <typename T>
class Wrapper_op : public Operation<T>
{
 public:
    explicit Wrapper_op(std::shared_ptr<Tensor_object<T>> input_tensor,
                        unsigned long operation_id,
                        const std::string &name = "constant");
    explicit Wrapper_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }
};

template <typename T>
class Operation_management
{
 public:
    static void add_op(Operation<T> operation);
    static void set_op(unsigned int id, const Operation<T> &operation);
    static void add_output_of(long id,
                              std::shared_ptr<Tensor_object<T>> tensor_ptr);
    static void add_input_of(long id,
                             std::shared_ptr<Tensor_object<T>> tensor_ptr);

    static void print_operation_info()
    {
        for (auto op : operation_list)
        {
            std::cout << op.print_info() << std::endl;
        }
    }

    static decltype(auto) get_operation_info(){

        std::vector<std::tuple<long, unsigned long, unsigned long>> op_vect;
        for (decltype(auto) operation : operation_list)
        {
                op_vect.emplace_back(operation.get_info());
        }

        return op_vect;
    }

    static unsigned long number_of_operations(){
        return operation_list.size();
    }

    static void create_adj(){
        Adj_management<T>::reserve_adj(operation_list.size());
        for(Operation<T> operation: operation_list){
            decltype(auto) input_tensor_vect = operation.get_input_tensor_vect();
            for(auto tensor_ptr: input_tensor_vect){
                Adj_management<T>::add_edge(tensor_ptr->get_from(), tensor_ptr->get_to(), tensor_ptr);
            }
        }
    }

 private:
    static Operation<T> &get_op(long operation_id)
    {
        for (decltype(auto) operation : operation_list)
        {
            if (operation.get_id() == operation_id)
                return operation;
        }
        // returns empty operation if nothing is found
        return operation_list[0];
    }
    static std::deque<Operation<T>> operation_list;
    static std::mutex operation_list_mutex;
};

template <typename T>
std::deque<Operation<T>> Operation_management<T>::operation_list =
    std::deque<Operation<T>>{ Empty_op<T>() };

template <typename T>
std::mutex Operation_management<T>::operation_list_mutex = std::mutex();

}  // namespace cubby_dnn

#endif