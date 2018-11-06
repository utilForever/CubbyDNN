//
// Created by Justin on 18. 11. 5.
//

#include "../../../../Includes/Backend/storage/Storage.h"

namespace cubby_dnn {

    template<typename T>
    Storage<T>::Storage(std::vector<T> &&data, std::vector<int> &&shape) noexcept {
        this->shape = std::forward<std::vector<int>>(shape);
        this->data = std::forward<std::vector<T>>(data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
    }

    template<typename T>
    Storage<T>::Storage(Storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
    }

    template<typename T>
    Storage<T> &Storage<T>::operator=(Storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
        return *this;
    }


}