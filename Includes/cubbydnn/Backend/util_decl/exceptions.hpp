//
// Created by jwkim on 18. 11. 6.
//

#ifndef CUBBYDNN_EXCEPTIONS_H
#define CUBBYDNN_EXCEPTIONS_H

#include <iostream>
#include <string>

namespace cubby_dnn
{
class TensorException : public std::runtime_error
{
 public:
    std::string message()
    {
        return msg;
    }

    TensorException() : std::runtime_error("TensorException")
    {
    }

    explicit TensorException(const std::string& msg)
        : std::runtime_error("TensorException")
    {
        this->msg = msg;
    }

 protected:
    std::string msg;
    std::string default_msg;
};

// TODO: Add more Exceptions if required

class ArgumentException : public TensorException
{
 public:
    ArgumentException()
    {
        default_msg = "Argument Exception";
        msg = default_msg;
        std::cout << msg << std::endl;
    }

    explicit ArgumentException(const std::string& msg)
    {
        this->msg = default_msg + msg;
        std::cout << msg << std::endl;
    }
};

class InvalidOperation : public TensorException
{
 public:
    InvalidOperation()
    {
        default_msg = "Invalid Operation";
        msg = default_msg;
        std::cout << msg << std::endl;
    }

    explicit InvalidOperation(const std::string& msg)
    {
        this->msg = default_msg + msg;
        std::cout << msg << std::endl;
    }
};

class EmptyObjectException : public TensorException
{
 public:
    EmptyObjectException()
    {
        default_msg = "Requesting object that doesn't exist";
        msg = default_msg;
        std::cout << msg << std::endl;
    }

    explicit EmptyObjectException(const std::string& msg)
    {
        this->msg = default_msg + msg;
        std::cout << msg << std::endl;
    }
};
}  // namespace cubby_dnn

#endif  // CUBBYDNN_EXCEPTIONS_H
