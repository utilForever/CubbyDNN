//
// Created by jwkim on 18. 11. 15.
//


#ifndef CUBBYDNN_STREAM_DECL_HPP
#define CUBBYDNN_STREAM_DECL_HPP

#include <vector>
#include <iostream>

template<typename T>
class stream{
public:
    stream() = default;

    virtual std::vector<T> next(){
        std::cout<<"Stream next() not implemented"<<std::endl;
        return std::vector<T>();
    };

    virtual bool has_next(){
        std::cout<<"Stream has_next() not implemented"<<std::endl;
        return false;
    };

    long get_stream_size() { return stream_size; }

private:
    long stream_size = 0;
};

template<typename T>
class file_stream: public stream<T>{
public:
    file_stream(){
        std::cout<<"file_stream not implemented"<<std::endl;
    }

    std::vector<T> next() override{
        std::cout<<"file_stream next() not implemented"<<std::endl;
        return std::vector<T>();

    }

    bool has_next() override{
        std::cout<<"file_stream has_next() not implemented"<<std::endl;
        return false;
    }
};

template<typename T>
class data_stream: public stream<T>{
    data_stream(){
        std::cout<<"data_stream not implemented"<<std::endl;
    }
    std::vector<T> next() override{
        std::cout<<"data_stream next() not implemented"<<std::endl;
        return std::vector<T>();
    }

    bool has_next() override{
        std::cout<<"data_stream has_next() not implemented"<<std::endl;
        return false;
    }

};

#endif //CUBBYDNN_STREAM_DECL_HPP
