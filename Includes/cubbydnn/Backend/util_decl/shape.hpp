//
// Created by Justin on 18. 11. 20.
//

#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <iostream>
#include <string>
#include <vector>

struct tensor_shape
{
 public:
    tensor_shape() = default;

    tensor_shape(long rows, long columns, long height);

    bool operator==(const tensor_shape &rhs) const;

    bool operator!=(const tensor_shape &rhs) const;

    long size() const
    {
        return total_size;
    }

    bool empty() const
    {
        return shape_vect.empty();
    }

    long rows() const
    {
        return shape_vect.at(0);
    }

    long cols() const
    {
        return shape_vect.at(1);
    }

    long height() const
    {
        return shape_vect.at(2);
    }

    const std::vector<long> &get_shape_vect()
    {
        return shape_vect;
    }

 private:
    std::vector<long> shape_vect;
    long total_size = 0;
};

struct shape
{
 public:
    static bool check_shape(const tensor_shape &shape,
                            const std::string &op_name = "constructor");
};

#endif  // CUBBYDNN_SHAPE_HPP
