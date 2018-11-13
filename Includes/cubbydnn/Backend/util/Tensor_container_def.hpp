//
// Created by jwkim on 18. 11. 13.
//

#ifndef CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
#define CUBBYDNN_TENSOR_CONTAINER_DEF_HPP

#include "Backend/util_decl/Tensor_container_decl.hpp"

namespace cubby_dnn{
    ///definitions

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> &shape, cubby_dnn::Tensor_type type,
                      const int from, std::string name) {
        this->shape = shape;
        this->type = type;
        this->from = from;
        this->name = name;
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> &&shape, cubby_dnn::Tensor_type type,
                      const int from, std::string name) noexcept{
        this->shape = std::forward<std::vector<int>>(shape);
        this->type = type;
        this->from = from;
        this->name = name;
    }

    template<typename T>
    struct Tensor_container_decl<T>::storage{
        std::vector<T> data; //stores actual data with data type 'T'
        std::vector<int> shape; //shape of the data
        typedef std::size_t size_type;
        size_type byte_size;
    };

    template<typename T>
    Tensor_container_decl<T>::Tensor_container(const std::vector<T> &data, const std::vector<int> &shape,
                                          Tensor_type type, std::string name, const int tensor_id): type(type) {

        verify<T>(data, shape); //throws exception if arguments are invalid

        this->tensor_object = std::make_unique<storage>(data, shape);
        this->name = name;
        this->id = tensor_id;
    }

    template<typename T>
    Tensor_container_decl<T>::Tensor_container(std::vector<T> &&data, std::vector<int> &&shape,
                                          Tensor_type type, std::string name, const int tensor_id) noexcept: type(type) {

        verify<T>(data, shape); //throws exception if arguments are invalid

        this->tensor_object = std::make_unique<storage>(std::forward<std::vector<T>>(data),
                                                        std::forward<std::vector<int>>(shape));
        this->name = name;
        this->id = tensor_id;
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
#endif //CUBBYDNN_TENSOR_CONTAINER_DEF_HPP
