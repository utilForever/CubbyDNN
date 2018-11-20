
#ifndef CUBBYDNN_BASE_OPERATIONS_DECL_HPP
#define CUBBYDNN_BASE_OPERATIONS_DECL_HPP

#include "Backend/util/Tensor_container.hpp"

namespace cubby_dnn
{
enum class operation_type
{
    generate,
    basic,
    final
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

    bool has_input_vector()
    {
        return !input_tensor_vect.empty();
    }

    bool has_output_vector()
    {
        return !output_tensor_vect.empty();
    }

 protected:
    explicit Operation();

 protected:
    operation_type op_type;
    unsigned long operation_id;
    std::vector<std::shared_ptr<Tensor_object<T>>> input_tensor_vect;
    std::vector<std::shared_ptr<Tensor_object<T>>> output_tensor_vect;
    std::string name;
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
                        const std::vector<int> &shape,
                        unsigned long operation_id,
                        const std::string &name = "reshape");

    explicit Reshape_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    std::vector<int> shape;
};

template <typename T>
class placeHolder_op : public Operation<T>
{
 public:
    explicit placeHolder_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                            const std::vector<int> &shape,
                            unsigned long operation_id,
                            const std::string &name = "placeHolder");

    explicit placeHolder_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    std::vector<int> shape;
};

template <typename T>
class weight_op : public Operation<T>
{
 public:
    explicit weight_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                       const std::vector<int> &shape,
                       unsigned long operation_id,
                       const std::string &name = "weight");

    explicit weight_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    std::vector<int> shape;
};

template <typename T>
class constant_op : public Operation<T>
{
 public:
    explicit constant_op(std::shared_ptr<Tensor_object<T>> output_tensor,
                         const std::vector<int> &shape,
                         unsigned long operation_id,
                         const std::string &name = "constant");

    explicit constant_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }  // empty constructor for operation

 private:
    std::vector<int> shape;
};

template <typename T>
class wrapper_op : public Operation<T>
{
 public:
    explicit wrapper_op(std::shared_ptr<Tensor_object<T>> input_tensor,
                        unsigned long operation_id,
                        const std::string &name = "constant");
    explicit wrapper_op(unsigned long operation_id, const std::string &name)
    {
        this->operation_id = operation_id;
        this->name = name;
    }
};

template <typename T>
class Operation_management
{
 public:
    void add_op(Operation<T> operation);
    void set_op(unsigned int id, const Operation<T> &operation);
    void add_output_of(long id, std::shared_ptr<Tensor_object<T>> tensor_ptr);

 private:
    Operation<T> &get_op(long operation_id)
    {
        return operation_list[operation_id];
    }
    std::deque<Operation<T>> operation_list;
    std::mutex operation_list_mutex;
};
}  // namespace cubby_dnn

#endif