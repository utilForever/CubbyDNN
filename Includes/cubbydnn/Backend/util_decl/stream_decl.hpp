//
// Created by jwkim on 18. 11. 15.
//


#ifndef CUBBYDNN_STREAM_DECL_HPP
#define CUBBYDNN_STREAM_DECL_HPP

#include <vector>

template<typename T>
class Stream{
    virtual std::vector<T> next();

    virtual bool has_next();
};

#endif //CUBBYDNN_STREAM_DECL_HPP
