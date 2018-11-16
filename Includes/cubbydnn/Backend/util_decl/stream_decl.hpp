//
// Created by jwkim on 18. 11. 15.
//


#ifndef CUBBYDNN_STREAM_DECL_HPP
#define CUBBYDNN_STREAM_DECL_HPP

#include <vector>

template<typename T>
class Stream{
public:
    virtual std::vector<T> next() = 0;

    virtual bool has_next() = 0;

    long get_stream_size() { return stream_size; }

private:
    long stream_size = 0;
};

template<typename T>
class File_stream: public Stream<T>{

};

template<typename T>
class Data_stream: public Stream<T>{

};

#endif //CUBBYDNN_STREAM_DECL_HPP
