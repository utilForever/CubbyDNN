//
// Created by Justin on 18. 11. 5.
//

#include <../../Includes/Backend/storage/storage.h>

namespace cubby_dnn {

    template<typename T>
    storage<T>::storage(std::vector<T> &&data, std::vector<int> &&shape) noexcept {
        this->shape = std::forward<std::vector<int>>(shape);
        this->data = std::forward<std::vector<T>>(data);
        this->size_type = sizeof(T);
    }

    template<typename T>
    storage<T>::storage(storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->size_type = sizeof(T);
    }

    template<typename T>
    storage<T> &storage<T>::operator=(storage<T> &&rhs) noexcept {
        this->shape = std::forward<std::vector<int>>(rhs.shape);
        this->data = std::forward<std::vector<T>>(rhs.data);
        this->size_type = sizeof(T);
        return *this;
    }


}