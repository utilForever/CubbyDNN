//
// Created by Justin on 18. 11. 5.
//

#include "Backend/util/storage.h"

namespace cubby_dnn {

    template<typename T>
    storage<T>::storage(std::vector<T> &&data, std::vector<int> &&shape) {
        //TODO: throw exception if shape and data type does not match
        this->shape = std::forward<std::vector<int>>(shape);
        this->data = std::forward<std::vector<T>>(data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
    }

    template<typename T>
    storage<T>::storage(storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
    }

    template<typename T>
    storage<T> &storage<T>::operator=(storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->byte_size = static_cast<size_type>(sizeof(T)) * this->data.size();
        return *this;
    }


}