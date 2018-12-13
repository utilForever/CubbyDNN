//
// Created by jwkim98 on 18. 12. 13.
//

#ifndef CUBBYDNN_ERROR_MANAGEMENT_HPP
#define CUBBYDNN_ERROR_MANAGEMENT_HPP

#include <string>

namespace cubby_dnn
{
enum class err_type
{
    memory_error,
    invalid_shape,
    shape_matching,
    not_implemented,
};

    std::ostream& operator<<(std::ostream& out, err_type value);

class terminal {
public:
    static void print_error(err_type type, const std::string& calling_method,
            const std::string& description);
private:
    static bool error_state ;
};

}  // namespace cubby_dnn

#endif  // CUBBYDNN_ERROR_MESSAGE_HPP
