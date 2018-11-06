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

        tensor(std::vector<T> &&data, std::vector<int> &&shape, std::string name);

        tensor(tensor<T> &&rhs) noexcept;

        tensor(const tensor<T> &rhs) noexcept;

        tensor& operator=(tensor &&rhs) noexcept;

        tensor& operator=(const tensor &rhs);

        virtual ~tensor() = default;

        virtual tensor operator+(const tensor &rhs);

        virtual tensor& operator+=(const tensor &rhs);

        virtual tensor operator-(const tensor &rhs);

        virtual tensor& operator-=(const tensor &rhs);

        virtual tensor operator*(const tensor<T> &rhs);

        virtual tensor& operator*=(const tensor<T> &rhs);

        virtual tensor matmul(const tensor<T> &rhs);

        virtual tensor& reshape();

        virtual tensor& transpose();

        const std::string &getName() const;

    protected:
        std::unique_ptr<storage<T>> Data; //Stores data required
        std::string name = nullptr;
    };
}

#endif //CUBBYDNN_BACKEND_H
