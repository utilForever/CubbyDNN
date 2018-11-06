//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_STORAGE_H
#define CUBBYDNN_STORAGE_H

#include <vector>

namespace cubby_dnn {

    template<typename T>
    class storage final {
    protected:
        storage(std::vector<T> &&data, std::vector<int> &&shape) noexcept;

        storage(storage<T> &&rhs) noexcept;

        storage<T> &operator=(storage<T> &&rhs) noexcept;

        std::vector<T> data; //stores actual data with data type 'T'
        std::vector<int> shape; //shape of the data
        std::size_t size_type;
    };

}

#endif //CUBBYDNN_STORAGE_H
