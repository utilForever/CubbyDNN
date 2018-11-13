//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_BACKEND_H
#define CUBBYDNN_BACKEND_H

#include <vector>
#include <memory>
#include "../../../../Includes/cubbydnn/Backend/util/exceptions.hpp"

namespace cubby_dnn {

    enum class Tensor_type{
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

        Tensor(std::vector<int> &shape, Tensor_type type, const int from, std::string name) noexcept;

        Tensor(std::vector<int> &&shape, Tensor_type type, const int from, std::string name) noexcept;


    public:
        ///getters
        const Tensor_type getType() const { return type; }

        const std::string& getName() const { return name; }

        const bool isTrainable() const { return trainable; }

    private:
        std::string name = nullptr;
        bool trainable = true;
        int from;
        int to = -1;
        Tensor_type type;
        std::vector<int> shape;
    };

    template<typename T>
    void verify(std::vector<T> &data, std::vector<int> &shape); //throws exception if input in invalid

    template<typename T>
    class Tensor_container{
    public:

        Tensor_container(std::vector<T> &data, std::vector<int> &shape, Tensor_type type, std::string name);

        Tensor_container(std::vector<T> &&data, std::vector<int> &&shape, Tensor_type type, std::string name);

        Tensor_container(const Tensor_container<T>& rhs);

        Tensor_container(Tensor_container<T>&& rhs) noexcept;

        Tensor_container& operator=(const Tensor_container<T>& rhs);

        Tensor_container& operator=(Tensor_container<T>&& rhs) noexcept;

        ~Tensor_container();

    private:

        struct storage;
        std::unique_ptr<storage> tensor_object; //container of actual data
        std::string name = nullptr;
        bool trainable = true;
        Tensor_type type;

        int op_from, op_to;

    public:
        ///getters
        const Tensor_type get_type() const { return type; };

        const std::string& get_name() const { return name; };

        const unsigned long get_data_size() const { return tensor_object->data.size(); };

        const std::vector<int>& get_shape() const { return tensor_object->data.shape(); };

        const std::vector<int>& get_data() const { return tensor_object->data; };

        const bool is_trainable() const { return trainable; }


        const int from() const { return op_from; }

        ///setters
        void disable_training() { trainable = false; }

        void enable_training() { trainable = true; }
    };
}

#endif //CUBBYDNN_BACKEND_H
