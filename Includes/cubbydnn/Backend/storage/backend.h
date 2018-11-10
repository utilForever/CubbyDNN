//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include "storage.h"
#include <memory>
#include "Backend/storage/exceptions.h"

namespace cubby_dnn {

    enum class TensorType{
        weight,
        bias,
        filter,
        other
    };

    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape); //throws exception if input in invalid

    template<typename T>
    class Tensor: private storage<T> {
    public:
        Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type);

        Tensor(std::vector<T> &data, std::vector<int> &shape, TensorType type, std::string name);

        Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type);

        Tensor(std::vector<T> &&data, std::vector<int> &&shape, TensorType type, std::string name);


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

        Tensor(Tensor<T> &&rhs) noexcept; //This will get ownership of storage

        Tensor(const Tensor<T> &rhs) noexcept; //This will copy the storage

        Tensor& operator=(Tensor &&rhs) noexcept;

        Tensor& operator=(const Tensor &rhs);

    public:
        ///getters
        const TensorType getType() const { return type; };

        const std::string& getName() const { return name; };

        const unsigned long getDataSize() const { return Data->data.size(); };

        const std::vector<int>& getShape() const { return Data->data.shape(); };

        const std::vector<int>& getData() const { return Data->data; };

        const bool isTrainable() const { return trainable; }

        ///setters
        void disableTraining() { trainable = false; }

        void enableTraining() { trainable = true; }
    };
}

#endif //CUBBYDNN_BACKEND_H
