//
// Created by jwkim on 18. 12. 4.
//

#ifndef CUBBYDNN_STREAM_HPP
#define CUBBYDNN_STREAM_HPP

#include "backend/util_decl/stream_decl.hpp"

namespace cubby_dnn
{
template <typename T>
std::vector<T> stream<T>::next()
{
    std::cout << "Stream next() not implemented" << std::endl;
    return std::vector<T>();
}

template <typename T>
bool stream<T>::has_next()
{
    std::cout << "Stream has_next() not implemented" << std::endl;
    return false;
}

template <typename T>
long stream<T>::get_stream_size()
{
    return stream_size;
}

template <typename T>
file_stream<T>::file_stream()
{
    std::cout << "file_stream not implemented" << std::endl;
}

template<typename T>
std::vector<T> file_stream<T>::next()
{
    std::cout << "file_stream next() not implemented" << std::endl;
    return std::vector<T>();
}

template<typename T>
bool file_stream<T>::has_next(){
    std::cout << "file_stream has_next() not implemented" << std::endl;
    return false;
}

template <typename T>
data_stream<T>::data_stream()
{
    std::cout << "data_stream not implemented" << std::endl;
}

template <typename T>
std::vector<T> data_stream<T>::next()
{
    std::cout << "data_stream next() not implemented" << std::endl;
    return std::vector<T>();
}

template <typename T>
bool data_stream<T>::has_next()
{
    std::cout << "data_stream has_next() not implemented" << std::endl;
    return false;
}

}  // namespace cubby_dnn

#endif  // CUBBYDNN_STREAM_HPP
