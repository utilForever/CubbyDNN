//
// Created by Justin on 18. 11. 5.
// Backend Base class for implementing basic tensor operations
//
#include <memory>
#include <vector>
#include "Backend/util/Tensor_container.hpp"
#include "Backend/util/exceptions.hpp"
#include "Backend/util/management.hpp"

namespace cubby_dnn {

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

}