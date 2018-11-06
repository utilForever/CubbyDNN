//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include "storage/Storage.h"
#include <memory>

namespace cubby_dnn {
    template<typename T>
    class Tensor: private Storage<T> {
    public:
        Tensor(std::vector<T> &&data, std::vector<int> &&shape);

        Tensor(std::vector<T> &&data, std::vector<int> &&shape, std::string name);

        Tensor(Tensor<T> &&rhs) noexcept;

        Tensor(const Tensor<T> &rhs) noexcept;

        Tensor& operator=(Tensor &&rhs) noexcept;

        Tensor& operator=(const Tensor &rhs);

        virtual ~Tensor() = default;

        virtual Tensor operator+(const Tensor &rhs); //Add

        virtual Tensor& operator+=(const Tensor &rhs);

        virtual Tensor operator-(const Tensor &rhs); //Sub

        virtual Tensor& operator-=(const Tensor &rhs);

        virtual Tensor operator*(const Tensor<T> &rhs); //Dot product

        virtual Tensor& operator*=(const Tensor<T> &rhs);

        virtual Tensor matmul(const Tensor<T> &rhs); //matrix multiplication

        virtual Tensor& reshape();

        virtual Tensor& transpose();

        const std::string &getName() const;

    protected:
        std::unique_ptr<Storage<T>> Data; //Stores data required
        std::string name = nullptr;
    };
}

#endif //CUBBYDNN_BACKEND_H
