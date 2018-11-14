//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <vector>
#include <memory>
#include "../../../../Includes/cubbydnn/Backend/util_decl/exceptions.hpp"

#include <vector>
#include <deque>
#include <mutex>

namespace cubby_dnn {

    enum class Tensor_type{
        weight,
        bias,
        filter,
        placeHolder,
        other
    };


    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape); //throws exception if input in invalid

    template<typename T>
    class Tensor_container{
    public:

        Tensor_container(const std::vector<T> &data, const std::vector<int> &shape, Tensor_type type, std::string name,
                int tensor_id);

        Tensor_container(std::vector<T> &&data, std::vector<int> &&shape, Tensor_type type, std::string name,
                int tensor_id) noexcept;

        Tensor_container(const Tensor_container<T>& rhs);

        Tensor_container(Tensor_container<T>&& rhs) noexcept;

        Tensor_container& operator=(const Tensor_container<T>& rhs);

        Tensor_container& operator=(Tensor_container<T>&& rhs) noexcept;

        ~Tensor_container();

    private:

        struct storage;
        std::unique_ptr<storage> tensor_object; //container of actual data
        std::string name = nullptr;
        bool trainable = true;
        Tensor_type type;
        int id;

    public:
        ///getters
        constexpr Tensor_type get_type() const { return type; }

        const std::string& get_name() const { return name; }

        constexpr int get_data_size() const { return static_cast<int>(tensor_object->data.size()); }

        constexpr int get_data_byte_size() const { return static_cast<int>(tensor_object->data.size()*sizeof(T)); }

        const std::vector<int>& get_shape() const { return tensor_object->data.shape(); }

        const std::vector<int>& get_data() const { return tensor_object->data; }

        constexpr bool is_trainable() const { return trainable; }

        ///setters
        void disable_training() { trainable = false; }

        void enable_training() { trainable = true; }
    };

    template<typename T>
    class Tensor{

    private:

        Tensor(std::vector<int> &shape, Tensor_type type, int tensor_id, int from, std::string name,
               bool _mutable = true);

        Tensor(std::vector<int> &&shape, Tensor_type type, int tensor_id, int from, std::string name,
               bool _mutable = true) noexcept;

    public:
        ///getters
        constexpr Tensor_type get_type() const { return type; }

        const std::string& get_name() const {
            if(tensor_container_ptr)
                return tensor_container_ptr->name;
            else
                throw EmptyObjectException();
        }

        constexpr bool is_mutable() const { return _mutable; }

        const std::weak_ptr<Tensor_container<T>> &get_tensor_container_ptr() const {
            return tensor_container_ptr;
        }

        //TODO: Actual modification on the Tensor_container required
        ///setters
        void set_type(Tensor_type type) { this->type = type; }

        void set_name(const std::string &name) { tensor_container_ptr->name = name; }

        void make_mutable() { this->_mutable = true; }

        void make_constant() { this->_mutable =  false; }

    private:;
        //properties of the tensor

        bool _mutable = true; // determines whether data of this tensor can be modified
        int id; // specific ID to identify the tensor
        int from; // ID of operation that this tensor is generated
        int to = -1; // ID of operation that receives this tensor
        Tensor_type type; // type of this tensor
        std::vector<int> shape; // shape of this tensor

        std::weak_ptr<Tensor_container<T>> tensor_container_ptr;

    };

/// Resource management

    template<typename T>
    class Management{
    public:
        ///Adds new operation
        static int add_op() noexcept;
        ///Adds new edge between two
        static void add_edge(int from, int to, Tensor_container<T> &tensor) noexcept;
        ///Adds placeholders that can stream data into the graph
        static void add_placeHolder(std::unique_ptr<Tensor_container<T>> placeHolder) noexcept;

        static int get_graph_size() { return static_cast<int>(adj_forward.size()); }

        static std::unique_ptr<Tensor_container<T>> get_tensor_ptr(int from, int to) noexcept;

    private:

        static constexpr int default_graph_size = 30;

        static std::deque<std::shared_ptr<Tensor_container<T>>> placeHolders;

        static std::deque<std::deque<std::shared_ptr<Tensor_container<T>>>> adj_forward;

        Management() = default; ///disable the constructor

        static std::mutex adj_mutex; //mutex for restricting access to adj matrix
    };

}

#endif //CUBBYDNN_BACKEND_H
