//
// Created by Justin on 18. 11. 15.
//

#ifndef CUBBYDNN_STREAM_DECL_HPP
#define CUBBYDNN_STREAM_DECL_HPP

#include <iostream>
#include <vector>

namespace cubby_dnn
{
template <typename T>
class stream
{
 public:
    stream() = default;

    virtual std::vector<T> next();

    virtual bool has_next();

    long get_stream_size();

 private:
    long stream_size = 0;
};

template <typename T>
class file_stream : public stream<T>
{
 public:
    file_stream();

    std::vector<T> next() override;

    bool has_next() override;
};

    template <typename T>
class data_stream : public stream<T>
{
    data_stream();

    std::vector<T> next() override;

    bool has_next() override;

};
}  // namespace cubby_dnn

#endif  // CUBBYDNN_STREAM_DECL_HPP
