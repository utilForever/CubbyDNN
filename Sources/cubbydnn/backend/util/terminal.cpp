//
// Created by Justin on 18. 12. 13.
//
#include "backend/util/terminal.hpp"
#include <iostream>

namespace cubby_dnn
{
bool terminal::error_state = false;

std::ostream& operator<<(std::ostream& out, const err_type value)
{
    std::string msg = "error_type : ";
    switch (value)
    {
        case err_type::memory_error:
            msg += "memory_error";
            break;
        case err_type::invalid_shape:
            msg += "invalid_shape";
            break;
        case err_type::shape_matching:
            msg += "shape_matching";
            break;
        case err_type::not_implemented:
            msg += "not_implemented";
    }
    return out << msg;
}

void terminal::print_error(err_type type,
                                   const std::string& calling_method,
                                   const std::string& description)
{
    std::cout << type << "\n"
              << calling_method << "\n"
              << description << std::endl;
}
}  // namespace cubby_dnn