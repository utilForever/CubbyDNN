//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <vector>
#include <memory>
#include "../../../../Includes/cubbydnn/Backend/util/exceptions.hpp"

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
    class Tensor{
    public:
        static constexpr int placeHolder_operation_index = -1;

        //TODO: think about ways to put data stream through placeholders

        static Tensor placeHolder(std::vector<int> shape);

        static Tensor placeHolder(std::vector<int> shape, std::string name);

        static Tensor weight(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor weight(std::vector<int> shape, bool trainable = true);

        static Tensor filter(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor filter(std::vector<int> shape, bool trainable = true);

    private:

        Tensor(std::vector<int> &shape, Tensor_type type, const int from, std::string name) noexcept;

        Tensor(std::vector<int> &&shape, Tensor_type type, const int from, std::string name) noexcept;

    public:
        ///getters
        const Tensor_type getType() const { return type; }

        const std::string& getName() const { return name; }

        const bool isTrainable() const { return trainable; }

    private:
        std::string name = nullptr;
        bool trainable = true;
        int from;
        int to = -1;
        Tensor_type type;
        std::vector<int> shape;
    };

    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape); //throws exception if input in invalid

    template<typename T>
    class Tensor_container{
    public:

        Tensor_container(std::vector<T> &data, std::vector<int> &shape, Tensor_type type, std::string name);

        Tensor_container(std::vector<T> &&data, std::vector<int> &&shape, Tensor_type type, std::string name);

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

        int op_from, op_to;

    public:
        ///getters
        const Tensor_type get_type() const { return type; };

        const std::string& get_name() const { return name; };

        const unsigned long get_data_size() const { return tensor_object->data.size(); };

        const std::vector<int>& get_shape() const { return tensor_object->data.shape(); };

        const std::vector<int>& get_data() const { return tensor_object->data; };

        const bool is_trainable() const { return trainable; }


        const int from() const { return op_from; }

        ///setters
        void disable_training() { trainable = false; }

        void enable_training() { trainable = true; }
    };

///management

    template<typename T>
    class Management{
    public:
        ///Adds new operation
        static int add_op() noexcept;
        ///Adds new edge between two
        static void add_edge(const int from, const int to, Tensor_container<T> &tensor) noexcept;
        ///Adds placeholders that can stream data into the graph
        static void add_placeHolder(std::unique_ptr<Tensor_container<T>> placeHolder) noexcept;

        static int get_graph_size(){
            return static_cast<int>(adj_forward.size());
        }

        static std::unique_ptr<Tensor_container<T>> get_tensor_ptr(const int from, const int to) noexcept;

    private:
        static std::deque<std::unique_ptr<Tensor_container<T>>> placeHolders;

        static std::deque<std::vector<std::unique_ptr<Tensor_container<T>>>> adj_forward;

        Management(){} ///disable the constructor

        static std::mutex adj_mutex;
    };



    template<typename T>
    Tensor<T> Tensor<T>::placeHolder(std::vector<int> shape){
        if(shape.size() == 0)
            throw ArgumentException("argument 'shape' is empty");
        else if(shape.size() > 3)
            throw ArgumentException("dimension of shape is over 3");

        int this_num = Management<T>::get_plalsceHolder_num();
        std::string name = "placeHolder{" + std::to_string(this_num +1 ) + "}";

        Tensor<T> tensor(shape, Tensor_type::placeHolder,
                         placeHolder_operation_index, name);

        unsigned long size = 1;
        for(auto elem : shape){
            size *= elem;
        }

        std::vector<T> data(size);

        Tensor_container<T> tensor_object(std::move(data), std::move(shape),
                                          Tensor_type::placeHolder, name);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

    template<typename T>
    Tensor<T> Tensor<T>::placeHolder(std::vector<int> shape, std::string name){

        Tensor<T> tensor(shape, Tensor_type::placeHolder, -1,
                         placeHolder_operation_index, name);

        unsigned long size = 1;
        for(auto elem : shape){
            size *= elem;
        }

        std::vector<T> data(size);

        Tensor_container<T> tensor_object(data, shape, Tensor_type::placeHolder, name);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> &shape, cubby_dnn::Tensor_type type,
                      const int from, std::string name) noexcept{
        this->shape = shape;
        this->type = type;
        this->from = from;
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> &&shape, cubby_dnn::Tensor_type type,
                      const int from, std::string name) noexcept{
        this->shape = std::forward<std::vector<int>>(shape);
        this->type = type;
        this->from = from;
    }

    template<typename T>
    struct Tensor_container<T>::storage{

        std::vector<T> data; //stores actual data with data type 'T'
        std::vector<int> shape; //shape of the data
        typedef std::size_t size_type;
        size_type byte_size;
    };


    template<typename T>
    Tensor_container<T>::Tensor_container(std::vector<T> &data, std::vector<int> &shape,
                                          Tensor_type type, std::string name): type(type) {

        verify<T>(data, shape); //throws exception if arguments are invalid

        this->tensor_object = std::make_unique<storage>(std::forward<std::vector<T>>(data),
                                                        std::forward<std::vector<int>>(shape));
        this->name = name;
    }

    template<typename T>
    Tensor_container<T>::Tensor_container(std::vector<T> &&data, std::vector<int> &&shape,
                                          Tensor_type type, std::string name): type(type) {

        verify<T>(data, shape); //throws exception if arguments are invalid

        this->tensor_object = std::make_unique<storage>(std::forward<std::vector<T>>(data),
                                                        std::forward<std::vector<int>>(shape));
        this->name = name;
    }

    template<typename T>
    Tensor_container<T>::Tensor_container(const Tensor_container<T> &rhs): tensor_object(nullptr) {

        if(rhs.tensor_object)
            this->tensor_object = std::make_unique<Tensor_container<T>::tensor_object>(*rhs.tensor_object);
    }

    template<typename T>
    Tensor_container<T>::Tensor_container(Tensor_container<T> &&rhs) noexcept = default;

    template<typename T>
    Tensor_container<T>& Tensor_container<T>::operator=(const cubby_dnn::Tensor_container<T> & rhs) {
        if(rhs.object)
            this->tensor_object = std::make_unique<Tensor_container<T>::tensor_object>(*rhs.tensor_object);
        return *this;
    }

    template<typename T>
    Tensor_container<T>& Tensor_container<T>::operator=(cubby_dnn::Tensor_container<T> &&rhs) noexcept = default;

    template<typename T>
    Tensor_container<T>::~Tensor_container() = default;


    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape){

        if(data.size() == 0)
            throw ArgumentException("empty data");

        unsigned long expected_size = 1;
        for(auto elem : shape){
            expected_size *= elem;
        }

        if(expected_size != data.size()) {
            std::string err_message = "data shape doesn't match";
            std::string expected = "Expected Size = " + std::to_string(expected_size);
            std::string given = "given data size = " + std::to_string(data.size());
            throw ArgumentException(err_message + expected + given);
        }
    }

/// management

    //TODO: make these thread-safe!
    template<typename T>
    int Management<T>::add_op() noexcept{
        unsigned long graph_size = adj_forward.size();
        std::vector<std::unique_ptr<Tensor_container<T>>> temp(graph_size + 1, nullptr);

        std::lock_guard<std::mutex> guard(adj_mutex);
        adj_forward.emplace_back(temp); // graph_size += 1
        return static_cast<int>(graph_size);
    }

    template<typename T>
    void Management<T>::add_edge(const int from, const int to, Tensor_container<T> &tensor) noexcept{

        try{
            int graph_size = static_cast<int>(adj_forward.size());
            if(from == to){
                std::string error_msg = "cannot connect to operation itself";
                throw ArgumentException(error_msg);
            }
            if(graph_size + 1 < from or graph_size+ 1 < to){
                std::string error_msg = "pointing to operation that doesn't exist";
                error_msg += ("graph size: " + std::to_string(adj_forward.size()) + "from: "
                              + std::to_string(from) + "to: " + std::to_string(to));
                throw ArgumentException(error_msg);
            }

            if(adj_forward[from][to] != nullptr){
                std::string error_msg = "this edge is already assigned";
                throw InvalidOperation(error_msg);
            }
        }
        catch(TensorException e){
            return; ///do nothing, and return
        }

        std::lock_guard<std::mutex> guard(adj_mutex);
        adj_forward[from][to] = make_unique(tensor);
    }

    template<typename T>
    std::unique_ptr<Tensor_container<T>> Management<T>::get_tensor_ptr(const int from, const int to) noexcept{
        try {
            if (from >= adj_forward.size() || to >= adj_forward.size()) {
                std::string error_msg = "pointing to operation that doesn't exist";
                error_msg += ("graph size: " + std::to_string(adj_forward.size()) + "from: "
                              + std::to_string(from) + "to: " + std::to_string(to));
                throw ArgumentException(error_msg);
            }
        }
        catch(TensorException e){
            return nullptr;
        }
        return adj_forward[from][to]; ///get ownership from adj (thread-safe);
    }

    template<typename T>
    void Management<T>::add_placeHolder(std::unique_ptr<Tensor_container<T>> placeHolder) noexcept{

        std::lock_guard<std::mutex> guard(adj_mutex);
        placeHolders.emplace_back(placeHolder);
    }
}

#endif //CUBBYDNN_BACKEND_H
