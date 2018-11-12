//
// Created by jwkim on 18. 11. 10.
//

#include "Backend/util/management.hpp"

namespace cubby_dnn{
    //TODO: make these thread-safe!
    template<typename T>
    int Management<T>::add_op() noexcept{
        unsigned long graph_size = adj_forward.size();
        std::vector<std::unique_ptr<Tensor_container<T>>> temp(graph_size + 1, nullptr);

        std::lock_guard<std::mutex> guard(adj_mutex);
        adj_forward.emplace_back(temp); // graph_size += 1
        return static_cast<int>(graph_size);
    }

    template<typename T>
    void Management<T>::add_edge(const int from, const int to, Tensor_container<T> &tensor) noexcept{

        try{
            int graph_size = static_cast<int>(adj_forward.size());
            if(from == to){
                std::string error_msg = "cannot connect to operation itself";
                throw ArgumentException(error_msg);
            }
            if(graph_size + 1 < from or graph_size+ 1 < to){
                std::string error_msg = "pointing to operation that doesn't exist";
                error_msg += ("graph size: " + std::to_string(adj_forward.size()) + "from: "
                              + std::to_string(from) + "to: " + std::to_string(to));
                throw ArgumentException(error_msg);
            }

            if(adj_forward[from][to] != nullptr){
                std::string error_msg = "this edge is already assigned";
                throw InvalidOperation(error_msg);
            }
        }
        catch(TensorException e){
            return; ///do nothing, and return
        }

        std::lock_guard<std::mutex> guard(adj_mutex);
        adj_forward[from][to] = make_unique(tensor);
    }

    template<typename T>
    std::unique_ptr<Tensor_container<T>> Management<T>::get_tensor_ptr(const int from, const int to) noexcept{
        try {
            if (from >= adj_forward.size() || to >= adj_forward.size()) {
                std::string error_msg = "pointing to operation that doesn't exist";
                error_msg += ("graph size: " + std::to_string(adj_forward.size()) + "from: "
                              + std::to_string(from) + "to: " + std::to_string(to));
                throw ArgumentException(error_msg);
            }
        }
        catch(TensorException e){
            return nullptr;
        }
        return adj_forward[from][to]; ///get ownership from adj (thread-safe);
    }

    template<typename T>
    void Management<T>::add_placeHolder(std::unique_ptr<Tensor_container<T>> placeHolder) noexcept{

        std::lock_guard<std::mutex> guard(adj_mutex);
        placeHolders.emplace_back(placeHolder);
    }
}