//
// Created by Justin on 18. 11. 5.
//

/**
 * @brief This class contains declaration of tensor and tensor_object class
 */

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include "backend/util_decl/shape.hpp"

namespace cubby_dnn
{
enum class tensor_type
{
    variable,
    normal,
    None
};

const static size_t error_id = 0;

template <typename T>
bool verify(const std::vector<T> &data, const tensor_shape &shape);

/**
 * @brief class for representing information and data of of the tensor
 *
 * This class will represent graph at runtime, and stores actual data used in
 * graph execution
 *
 * @tparam T : basic data type for operation
 */
template <typename T>
class tensor_data
{
 public:
    tensor_data(size_t data_size, const tensor_shape &shape, long from,
                long to);

    tensor_data(size_t data_size, tensor_shape &&shape, long from, long to);

    tensor_data(const tensor_data<T> &rhs);

    tensor_data(tensor_data<T> &&rhs) noexcept;

    tensor_data &operator=(const tensor_data<T> &rhs);

    tensor_data &operator=(tensor_data<T> &&rhs) noexcept;

    ~tensor_data();

 private:
    struct data;

 public:
    /// getters
    bool has_data() const;

    size_t get_data_size() const;

    size_t get_data_byte_size() const;

    const std::vector<T> get_data_vector() const;

    std::unique_ptr<data> import_tensor_storage();

    void return_tensor_storage(std::unique_ptr<typename tensor_data<T>::data> rhs);

    tensor_shape get_data_shape() const;

    bool is_mutable() const;

    void set_mutable();

    void set_constant();

    long comes_from() const;

    long heads_to() const;

 private:
    bool _mutable = true;

    bool busy = false;

    long from, to;
    /// tensor_storage points to actual data stored.
    std::unique_ptr<data> tensor_storage;

    std::mutex lock_tensor_storage;
};

/**
 * @brief Helper class shown to user for constructing graph.
 *
 * This graph contains information of graph being built
 * methods of operation class builds tensor_data class based on information of
 * this class
 *
 * @tparam T
 */
template <typename T>
class tensor
{
 public:
    tensor(const tensor_shape &shape, long from, bool _mutable = true);

 public:
    /// getters

    bool is_valid() const;

    const tensor_shape &get_shape() const;

    size_t get_data_size() const;

    bool is_mutable() const;

    long get_from() const;

    void make_mutable();

    void make_constant();

    void add_to(long to);

 private:
    long from;  /// ID of operation that this tensor is generated

    std::vector<long>
        to_vector;  /// vector for storing operations this tensor will head to

    bool _mutable =
        true;  /// determines whether data of this tensor can be modified

    tensor_shape shape;  /// shape of this tensor represents
};

}  // namespace cubby_dnn

#endif  // CUBBYDNN_BACKEND_H
