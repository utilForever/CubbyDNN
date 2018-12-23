
/**
 * Created by Justin on 18. 11. 13.
 *
 * @brief This file declares template classes that adds operations to the graph
 *
 * There are 3 types of class templates
 * These class templates will provide operations that can be executed after
 *compilation of the graph
 *
 * 1. generate
 * >member functions of this class template template will generate tensors
 *
 * 2. operate
 * >member functions of this class template will receive tensors from other
 *operations, and return the result as a single tensor
 *
 * 3. final
 * >member functions of this class template will receive tensors from other
 *operations, and will not return any. every tensor should be connected to this
 *final operation to be executed
 *
 */

#ifndef CUBBYDNN_GENERATE_TENSOR_HPP
#define CUBBYDNN_GENERATE_TENSOR_HPP

#include "backend/graph/operations.hpp"
#include "backend/graph/tensor.hpp"
#include "backend/management/graph_management.hpp"
#include "backend/util/terminal.hpp"

namespace cubby_dnn
{
/**
 * class template for adding operations that generates new tensors
 * @tparam T type of data
 */
template <typename T>
class generate
{
 public:
    // TODO: think about initialization methods
    enum class initializer
    {
        default_state
    };

    // TODO: think about ways to put data stream through placeholders

    /**
     * Generates tensor for placeholder that streams data from external source
     *
     * @param shape shape of the output tensor
     * @param stream stream object that will provide data
     * @param name name of this operation (default: "placeholder")
     * @return tensor<T> that will contain streamed data in runtime
     */
    static tensor<T> placeholder(const tensor_shape &shape, stream<T> &stream,
                                 const std::string &name = "placeholder");

    /**
     * Generates tensor for variable that can contain trainable weights, or
     * non-modifiable constants
     * @param shape shape of the output tensor
     * @param trainable true if tensor is trainable, false if tensor is constant
     * @param name name of this operation (default: "weight")
     * @return tensor<T> with desired preferences
     */
    static tensor<T> variable(const tensor_shape &shape, bool trainable = true,
                              const std::string &name = "weight");

 private:
    /// method for returning empty tensor
    /// @return empty tensor
    static tensor<T> get_default_tensor()
    {
        return tensor<T>(tensor_shape(), -1, "default Tensor due to error");
    }
};

/**
 * class template for adding operations that executes given tensors
 * @tparam T
 */
template <typename T>
class operate : protected tensor<T>
{
 public:
    operate() = default;

    /**
     * @brief inserts operation that multiplies given tensors to the graph
     * column size of first tensor and row size of second tensor should match
     * dimension of first tensor and second tensor should match
     * @param tensor1 first tensor to contain data for operation
     * @param tensor2 second tensor to contain data for operation
     * @param name name of the operation
     * @return tensor<T> to contain the result of the operation
     */
    static tensor<T> mat_mul(tensor<T> &tensor1, tensor<T> &tensor2,
                             const std::string &name = "mat_mul");

    /**
     * @brief inserts operation that adds given tensors to the graph
     * input tensors should have same shape
     * @param tensor1 first tensor to contain data for operation
     * @param tensor2 second tensor to contain data for operation
     * @param name name of the operation
     * @return tensor<T> to contain the result of the operation
     */
    static tensor<T> mat_add(tensor<T>& tensor1, tensor<T>& tensor2,
            const std::string& name = "mat_add");

    /**
     * @brief inserts operation that adds given tensors to the graph
     * applies dot-product
     * @param tensor1 tensor to contain data for operation
     * @param multiplier
     * @param name name of the operation
     * @return tensor<T> to contain the result of the operation
     */
    static tensor<T> mat_dot(tensor<T> &tensor1, T multiplier,
                             const std::string &name = "mat_dot");

    /**
     * @brief inserts operation that reshapes given tensor to the graph
     *
     * @param tensor1 tensor to reshape
     * @param shape shape of output tensor
     * @param name name of the operation
     * @return tensor<T> to contain the result of the operation
     */
    static tensor<T> reshape(tensor<T> &tensor1, const tensor_shape &shape,
                             const std::string &name = "reshape");

    static tensor<T> one_hot(tensor<T> &tensor1, size_t size,
                             const std::string &name = "one_hot");

 private:
    /// default tensor to return for handling errors
    static tensor<T> get_default_tensor();
};

/**
 * @brief final class template provides methods to mark that given tensor has no
 * more operation to be done
 * @tparam T
 */
template <typename T>
class final
{
 public:
    /**
     * inserts operation that marks given tensor has no more further operation
     * @param tensor1 tensor to mark as 'finished'
     * @param name name of the tensor
     */
    static void wrapper(tensor<T> &tensor1,
                        const std::string &name = "wrapper");
};
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATE_TENSOR_HPP