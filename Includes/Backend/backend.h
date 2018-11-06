//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <../../Includes/Backend/storage/storage.h>
#include <memory>

namespace cubby_dnn {
    template<typename T>
    class tensor: private storage<T> {
    public:
        tensor(std::vector<T> &&data, std::vector<int> &&shape);

        tensor(tensor<T> &&rhs) noexcept;

        tensor(const tensor<T> &rhs);

        tensor& operator=(tensor &&rhs) noexcept;

        tensor& operator=(const tensor &rhs);

        tensor& operator+(const tensor &rhs);

        tensor& operator+(const tensor &&rhs);

        virtual tensor& reshape();

        virtual tensor& transpose();

    protected:
        std::unique_ptr<storage<T>> Data; //Stores data required
    };
}

#endif //CUBBYDNN_BACKEND_H
