//
// Created by jwkim on 18. 11. 6.
//

#ifndef CUBBYDNN_EXCEPTIONS_H
#define CUBBYDNN_EXCEPTIONS_H

#include <string>

namespace cubby_dnn{
    class exception{
    public:
        exception(std::string msg){
            this.msg = msg;
        }
        std::string msg;
    };
}

#endif //CUBBYDNN_EXCEPTIONS_H
