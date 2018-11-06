//
// Created by Justin on 18. 11. 5.
// Backend Base class for implementing basic tensor operations
//
#include <memory>
#include <vector>
#include <../../Includes/Backend/backend.h>

namespace cubby_dnn {
    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &&data, std::vector<int> &&shape) {
        //TODO: Throw appropriate exception if data size and shape doesn't match
        this->Data = std::make_unique<Storage<T>>(std::forward<std::vector<T>>(data),
                std::forward<std::vector<int>>(shape));
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<T> &&data, std::vector<int> &&shape, std::string name) {
        //TODO: Throw appropriate exception if data size and shape doesn't match
        this->Data = std::make_unique<Storage<T>>(std::forward<std::vector<T>>(data),
                std::forward<std::vector<int>>(shape));
        this->name = name;
    }

    //This will transfer ownership to *this
    template<typename T>
    Tensor<T>::Tensor(Tensor<T> &&rhs) noexcept {
        this->Data = std::move(rhs.Data);
        rhs.data = nullptr; //explicitly make r-value data to null
    }

    //This will copy resources from rhs
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &rhs) noexcept{
        this->Data = std::make_unique<Storage<T>>(rhs.Data);
    }

    //transfers ownership to *this
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(cubby_dnn::Tensor<T> &&rhs) noexcept{
        this->Data = std::move(rhs.Data);
        rhs.data = nullptr;
        return *this;
    }

    //copies Data
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const cubby_dnn::Tensor<T> &rhs) {
        this->Data = std::make_unique<Storage<T>>(rhs.Data);
        return *this;
    }

    //returns name of this Tensor
    template<typename T>
    const std::string &Tensor<T>::getName() const {
        return name;
    }

}