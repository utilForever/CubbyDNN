//
// Created by Justin on 18. 11. 5.
//

#ifndef CUBBYDNN_STORAGE_H
#define CUBBYDNN_STORAGE_H

#include <vector>

namespace cubby_dnn {

    template<typename T>
    class Storage final {
    protected:
        Storage(std::vector<T> &&data, std::vector<int> &&shape) noexcept;

        Storage(Storage<T> &&rhs) noexcept;

        Storage<T> &operator=(Storage<T> &&rhs) noexcept;

        std::vector<T> data; //stores actual data with data type 'T'
        std::vector<int> shape; //shape of the data
        typedef std::size_t size_type;
        size_type byte_size;
    };

}

#endif //CUBBYDNN_STORAGE_H
