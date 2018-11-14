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
    class Tensor_object{
    public:

        Tensor_object(const std::vector<T> &data, const std::vector<int> &shape, Tensor_type type,
                         const std::string &name,
                         int tensor_id); //(1)

        Tensor_object(std::vector<T> &&data, std::vector<int> &&shape, Tensor_type type, std::string &&name,
                         int tensor_id); //(2)

        Tensor_object(const Tensor_object<T>& rhs); //(3)

        Tensor_object(Tensor_object<T>&& rhs) noexcept; //(4)

        Tensor_object& operator=(const Tensor_object<T>& rhs); //(5)

        Tensor_object& operator=(Tensor_object<T>&& rhs) noexcept; //(6)

        ~Tensor_object();

        /*
          for (4), (6) no exceptions can be thrown
          for (3), (5) std::bad_alloc may be thrown
          for (1), (2) if given _data_ or _shape_ is empty, _cubby_dnn::ArgumentException_ is thrown
          if given _shape_ does not match size of the data, _cubby_dnn::ArgumentException_ is thrown
         */

    private:

        std::string name = nullptr;

        bool _mutable = true;

        Tensor_type type;

        int tensor_id;

        struct storage;

        std::unique_ptr<storage> tensor_object; //container of actual data


    public:
        ///getters
        constexpr Tensor_type get_type() const { return type; }

        const std::string& get_name() const { return name; }

        constexpr int get_data_size() const { return static_cast<int>(tensor_object->data.size()); }

        constexpr long get_data_byte_size() const { return static_cast<long>(tensor_object->data.size()*sizeof(T)); }

        const std::vector<int>& get_shape() const { return tensor_object->data.shape(); }

        const std::vector<int>& get_data() const { return tensor_object->data; }

        constexpr int get_tensor_id() const { return tensor_id; }

        constexpr bool is_mutable() const { return _mutable; }

        ///setters
        void disable_training() { _mutable = false; }

        void enable_training() { _mutable = true; }
        
        void set_name(std::string name){ this->name = name; }

        void set_type(Tensor_type type){ this->type = type; }

        void make_mutable(){ this->_mutable = true; }

        void make_constant(){ this->_mutable = false; }
        
        //void set_id(int id){ this->tensor_id = id; }
    };

    template<typename T>
    class Tensor{

    private:

        Tensor(Tensor_type type, int from, bool _mutable = true); //(1)

    public:
        ///getters
        bool has_tensor_object(){
            return !tensor_container_ptr.expired();
        }

        constexpr Tensor_type get_type() const { return type; }

        const std::string& get_name() const {
            if(tensor_container_ptr)
                return tensor_container_ptr->name;
            else
                throw EmptyObjectException();
        }

        constexpr bool is_mutable() const { return _mutable; }

        const std::weak_ptr<Tensor_object<T>> &get_tensor_container_ptr() const {
            return tensor_container_ptr;
        }

        //TODO: think about ways to check if tensor_container_ptr is empty

        ///setters
        void set_tensor_object(std::weak_ptr<Tensor_object<T>> ptr){
            tensor_container_ptr = ptr;
        }

        void set_type(Tensor_type type) {
            this->type = type;
            if(tensor_container_ptr)
                tensor_container_ptr->set_type(type);
        }

        void set_name(const std::string &name) { 
            if(tensor_container_ptr)
                tensor_container_ptr->set_name(name);
        }

        void make_mutable() {
            this->_mutable = true;
            if(tensor_container_ptr)
                tensor_container_ptr->make_mutable();
        }

        void make_constant() {
            this->_mutable = false;
            if (tensor_container_ptr)
                tensor_container_ptr->make_constant();
        }

    private:
        //properties of the tensor
        bool _mutable = true; // determines whether data of this tensor can be modified

        int from; // ID of operation that this tensor is generated

        int to = -1; // ID of operation that receives this tensor

        Tensor_type type; // type of the tensor_container it is pointing to

        std::weak_ptr<Tensor_object<T>> tensor_container_ptr;
    };

/// Resource management

    template<typename T>
    class Management{
    public:
        ///Adds new operation
        static int add_op() noexcept;
        ///Adds new edge between two
        static void add_edge(int from, int to, Tensor_object<T> &tensor) noexcept;
        ///Adds placeholders that can stream data into the graph
        static void add_placeHolder(std::unique_ptr<Tensor_object<T>> placeHolder) noexcept;

        static int get_graph_size() { return static_cast<int>(adj_forward.size()); }

        static std::unique_ptr<Tensor_object<T>> get_tensor_ptr(int from, int to) noexcept;

    private:

        static constexpr int default_graph_size = 30;

        static std::deque<std::shared_ptr<Tensor_object<T>>> placeHolders;

        static std::deque<std::deque<std::shared_ptr<Tensor_object<T>>>> adj_forward;

        Management() = default; ///disable the constructor

        static std::mutex adj_mutex; //mutex for restricting access to adj matrix
    };

}

#endif //CUBBYDNN_BACKEND_H
