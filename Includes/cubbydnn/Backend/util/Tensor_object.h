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
        placeHolder,
        other
    };

    template<typename T>
    class Tensor{
    public:
        static constexpr int placeHolder_operation_index = -1;

        //TODO: think about ways to put data stream through placeholders

        static Tensor placeHolder(std::vector<int> shape);

        static Tensor placeHolder(std::vector<int> shape, std::string name);

        static Tensor weight(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor weight(std::vector<int> shape, bool trainable = true);

        static Tensor filter(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor filter(std::vector<int> shape, bool trainable = true);

    private:

        Tensor(std::vector<int> &shape, TensorType type, const int from, std::string name) noexcept;

        Tensor(std::vector<int> &&shape, TensorType type, const int from, std::string name) noexcept;


    public:
        ///getters
        const TensorType getType() const { return type; }

        const std::string& getName() const { return name; }

        const bool isTrainable() const { return trainable; }

    private:
        std::string name = nullptr;
        bool trainable = true;
        int from;
        int to = -1;
        TensorType type;
        std::vector<int> shape;
    };

    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape); //throws exception if input in invalid

    template<typename T>
    class Tensor_object: private storage<T> {
    public:

        Tensor_object(std::vector<T> &data, std::vector<int> &shape, TensorType type, std::string name);

        Tensor_object(std::vector<T> &&data, std::vector<int> &&shape, TensorType type, std::string name);


    private:
        std::unique_ptr<storage<T>> Data; //container of actual data
        std::string name = nullptr;
        bool trainable = true;
        TensorType type;

        ///Copy constructors are not allowed
        Tensor_object(const Tensor_object<T> &rhs) noexcept;

        ///Copy assignments are not allowed
        Tensor_object& operator=(Tensor_object &rhs) noexcept;

        int op_from, op_to;

    public:
        ///getters
        const TensorType getType() const { return type; };

        const std::string& getName() const { return name; };

        const unsigned long getDataSize() const { return Data->data.size(); };

        const std::vector<int>& getShape() const { return Data->data.shape(); };

        const std::vector<int>& getData() const { return Data->data; };

        const bool isTrainable() const { return trainable; }

        const int from() const { return op_from; }

        ///setters
        void disableTraining() { trainable = false; }

        void enableTraining() { trainable = true; }
    };
}

#endif //CUBBYDNN_BACKEND_H
