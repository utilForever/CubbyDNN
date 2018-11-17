
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
 protected:
    explicit Operation();

 protected:
    operation_type op_type;
    unsigned long operation_id;
    std::vector<std::shared_ptr<Tensor<T>>> input_tensor_vect;
    std::vector<std::shared_ptr<Tensor<T>>> output_tensor_vect;
};

template <typename T>
class Mat_mul_op : public Operation<T>
{
 public:
    explicit Mat_mul_op(std::shared_ptr<Tensor<T>> tensor1,
                        std::shared_ptr<Tensor<T>> tensor2,
                        std::shared_ptr<Tensor<T>> output_tensor);

    explicit Mat_mul_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect,
                        std::shared_ptr<Tensor<T>> output_tensor);
};

template <typename T>
class Mat_add_op : public Operation<T>
{
 public:
    explicit Mat_add_op(std::shared_ptr<Tensor<T>> tensor1,
                        std::shared_ptr<Tensor<T>> tensor2,
                        std::shared_ptr<Tensor<T>> output_tensor);

    explicit Mat_add_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect,
                        std::shared_ptr<Tensor<T>> output_tensor);
};

template <typename T>
class Mat_dot_op : public Operation<T>
{
 public:
    explicit Mat_dot_op(std::shared_ptr<Tensor<T>> tensor1,
                        std::shared_ptr<Tensor<T>> identity_tensor,
                        std::shared_ptr<Tensor<T>> output_tensor);
};

template <typename T>
class Reshape_op : public Operation<T>
{
 public:
    explicit Reshape_op(std::shared_ptr<Tensor<T>> tensor1,
                        std::shared_ptr<Tensor<T>> output_tensor,
                        const std::vector<int> &shape);

 private:
    std::vector<int> shape;
};

template <typename T>
class placeHolder_op : public Operation<T>
{
 public:
    explicit placeHolder_op(std::shared_ptr<Tensor<T>> output_tensor,
                            const std::vector<int> &shape);

 private:
    std::vector<int> shape;
};

template <typename T>
class weight_op : public Operation<T>
{
 public:
    explicit weight_op(std::shared_ptr<Tensor<T>> output_tensor,
                       const std::vector<int> &shape);

 private:
    std::vector<int> shape;
};

template <typename T>
class constant_op : public Operation<T>
{
 public:
    explicit constant_op(std::shared_ptr<Tensor<T>> output_tensor,
                         const std::vector<int> &shape);

 private:
    std::vector<int> shape;
};

template <typename T>
class Operation_management
{
 public:
    void add_op(const Operation <T> &operation, unsigned long id);

 private:
    std::deque<Operation<T>> operation_list;
    std::mutex operation_list_mutex;
};
}  // namespace cubby_dnn

#endif