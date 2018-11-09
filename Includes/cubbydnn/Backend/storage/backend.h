//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include "storage.h"
#include <memory>

namespace cubby_dnn {

    enum TensorType{
        weight,
        bias,
        filter
    };

    template<typename T>
    class Tensor: private storage<T> {
    public:
        Tensor(storage<T> storage);

        Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type);

        Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type, std::string name);

        Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type);

        Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type, std::string name);

        Tensor(Tensor<T> &&rhs) noexcept; //This will get ownership of storage

        Tensor(const Tensor<T> &rhs) noexcept; //This will copy the storage

        Tensor& operator=(Tensor &&rhs) noexcept;

        Tensor& operator=(const Tensor &rhs);


//        virtual ~Tensor();
//
//        virtual Tensor operator+(const Tensor &rhs); //Add
//
//        virtual Tensor& operator+=(const Tensor &rhs);
//
//        virtual Tensor operator-(const Tensor &rhs); //Sub
//
//        virtual Tensor& operator-=(const Tensor &rhs);
//
//        virtual Tensor operator*(const Tensor<T> &rhs); //Dot product
//
//        virtual Tensor& operator*=(const Tensor<T> &rhs);
//
//        virtual Tensor matmul(const Tensor<T> &rhs); //matrix multiplication
//
//        virtual Tensor reshape(const std::vector<int> shape);
//
//        virtual Tensor transpose();

    private:
        std::unique_ptr<storage<T>> Data; //Stores data required
        std::string name = nullptr;
        bool trainable = true;
        TensorType type;

    public:
        //getters
        const TensorType getType() const { return type; };

        const std::string& getName() const {return name; };

        const unsigned long getDataSize() const {return Data->data.size(); };

        const std::vector<int>& getShape() const { return Data->shape; };

        const std::vector<int>& getData() const { return Data->data; };

        const bool isTrainable() const { return trainable; }

        void disableTraining() { trainable = false; }

        void enableTraining() { trainable = true; }


    private:
        int prevOp = -1;
        int nextOp = -1;
    };
}

#endif //CUBBYDNN_BACKEND_H
