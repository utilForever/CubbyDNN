//
// Created by Justin on 18. 11. 5.
// Backend Base class for implementing basic tensor operations
//
#include <memory>
#include <vector>
#include <../../Includes/Backend/backend.h>

namespace cubby_dnn {
    template<typename T>
    tensor<T>::tensor(std::vector<T> &&data, std::vector<int> &&shape) {
        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                std::forward<std::vector<int>>(shape));
    }

    template<typename T>
    tensor<T>::tensor(std::vector<T> &&data, std::vector<int> &&shape, std::string name){
        this->Data = std::make_unique<storage<T>>(std::forward<std::vector<T>>(data),
                                                  std::forward<std::vector<int>>(shape));
        this->name = name;
    }

    //This will transfer ownership to *this
    template<typename T>
    tensor<T>::tensor(tensor<T> &&rhs) noexcept {
        this->Data = std::move(rhs.Data);
        rhs.data = nullptr; //explicitly make r-value data to null
    }

    //This will copy resources from rhs
    template<typename T>
    tensor<T>::tensor(const tensor<T> &rhs) noexcept{
        this->Data = std::make_unique<storage<T>>(rhs.Data);
    }

    //transfers ownership to *this
    template<typename T>
    tensor<T>& tensor<T>::operator=(cubby_dnn::tensor<T> &&rhs) noexcept{
        this->Data = std::move(rhs.Data);
        rhs.data = nullptr;
        return *this;
    }

    //copies Data
    template<typename T>
    tensor<T>& tensor<T>::operator=(const cubby_dnn::tensor<T> &rhs) {
        this->Data = std::make_unique<storage<T>>(rhs.Data);
        return *this;
    }

    template<typename T>
    const std::string &tensor<T>::getName() const {
        return name;
    }

}