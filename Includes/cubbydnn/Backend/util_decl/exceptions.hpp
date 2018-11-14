//
// Created by jwkim on 18. 11. 6.
//

#ifndef CUBBYDNN_EXCEPTIONS_H
#define CUBBYDNN_EXCEPTIONS_H

#include <string>
#include <iostream>

namespace cubby_dnn{
    
class TensorException: public std::exception{
    public:
        TensorException() = default;

        std::string message(){
            return msg;
        }

    protected:
        std::string msg;
        std::string default_msg;
    };

    //TODO: Add more Exceptions if required

    class ArgumentException: public TensorException{
    public:
        ArgumentException(){
            default_msg = "Argument Exception";
            msg = default_msg;
            std::cout<<msg<<std::endl;
        }

        explicit ArgumentException(const std::string& msg) {
            this->msg = default_msg + msg;
            std::cout<<msg<<std::endl;
        }
    };

    class InvalidOperation: public TensorException{
    public:
        InvalidOperation(){
            default_msg = "Invalid Operation";
            msg = default_msg;
            std::cout<<msg<<std::endl;
        }

        explicit InvalidOperation(const std::string& msg){
            this->msg = default_msg + msg;
            std::cout<<msg<<std::endl;
        }

    };
}

#endif //CUBBYDNN_EXCEPTIONS_H
