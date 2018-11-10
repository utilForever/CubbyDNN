//
// Created by Justin on 18. 11. 5.
// Backend Base class for implementing basic tensor operations
//
#include <memory>
#include <vector>
#include "Backend/storage/backend.h"
#include "Backend/storage/exceptions.h"

namespace cubby_dnn {

    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type): type(type){
        verify<T>(data, shape); //throws exception if arguments are invalid

        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                                                  std::forward<std::vector<int>>(shape));
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type, std::string name): type(type) {
        verify<T>(data, shape); //throws exception if arguments are invalid

        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                                                  std::forward<std::vector<int>>(shape));
        this->name = name;
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type): type(type) {
        verify<T>(data, shape); //throws exception if arguments are invalid


        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                std::forward<std::vector<int>>(shape));
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type, std::string name): type(type) {
        verify<T>(data, shape); //throws exception if arguments are invalid

        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                std::forward<std::vector<int>>(shape));
        this->name = name;
    }

//    //This will transfer ownership to *this
//    template<typename T>
//    Tensor<T>::Tensor(Tensor<T> &&rhs) noexcept {
//
//        this->Data = std::move(rhs.Data);
//        rhs.data = nullptr; //explicitly make r-value data to null
//    }
//
//    //This will copy resources from rhs
//    template<typename T>
//    Tensor<T>::Tensor(const Tensor<T> &rhs) noexcept {
//        this->Data = std::make_unique<storage<T>>(rhs.Data);
//    }
//
//    //transfers ownership to *this
//    template<typename T>
//    Tensor<T>& Tensor<T>::operator=(cubby_dnn::Tensor<T> &&rhs) noexcept {
//        this->Data = std::move(rhs.Data);
//        rhs.data = nullptr;
//        return *this;
//    }
//
//    //copies Data
//    template<typename T>
//    Tensor<T>& Tensor<T>::operator=(const cubby_dnn::Tensor<T> &rhs) {
//        this->Data = std::make_unique<storage<T>>(rhs.Data);
//        return *this;
//    }

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