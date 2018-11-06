//
// Created by jwkim on 18. 11. 6.
//

#ifndef CUBBYDNN_EXCEPTIONS_H
#define CUBBYDNN_EXCEPTIONS_H

#include <string>

namespace cubby_dnn{
    class TensorException{
    public:

        TensorException() = default;

        std::string msg;
    };

    //TODO: Add more Exceptions if required
    class ArgumentException: public TensorException{
    public:
        ArgumentException(){
            msg = "Argument Exception";
        }
    };

    class InvalidOperation: public TensorException{
        InvalidOperation(){
            msg = "Invalid Operation";
        }

    };
}

#endif //CUBBYDNN_EXCEPTIONS_H
