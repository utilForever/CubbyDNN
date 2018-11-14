
#Code Structure

Graph generation will be single-threaded
However, analyzing the graph and executing the graph can be multi-threaded

* ###class Tensor_container
    * This class is used to express tensor object on the user graph-building interface
    * ####constructors
    
    ````cpp
    
            Tensor_container(const std::vector<T> &data, const std::vector<int> &shape, Tensor_type type,
                             const std::string &name,
                             int tensor_id); //(1)
    
            Tensor_container(std::vector<T> &&data, std::vector<int> &&shape, Tensor_type type, std::string &&name,
                             int tensor_id); //(2)
    
            Tensor_container(const Tensor_container<T>& rhs); //(3)
    
            Tensor_container(Tensor_container<T>&& rhs) noexcept; //(4)
    
            Tensor_container& operator=(const Tensor_container<T>& rhs); //(5)
    
            Tensor_container& operator=(Tensor_container<T>&& rhs) noexcept; //(6)
    
            ~Tensor_container();
    ````
    
     * ####Exception safety
      for (4), (6) no exceptions can be thrown     
      for (3), (5) std::bad_alloc may be thrown     
      for (1), (2) if given _data_ or _shape_ is empty, _cubby_dnn::ArgumentException_ is thrown     
      if given _shape_ does not match size of the data, _cubby_dnn::ArgumentException_ is thrown
      
     * ####getters
     ````cpp
             constexpr Tensor_type get_type() const { return type; }
     
             const std::string& get_name() const { return name; }
     
             constexpr int get_data_size() const { return static_cast<int>(tensor_object->data.size()); }
     
             constexpr int get_data_byte_size() const { return static_cast<int>(tensor_object->data.size()*sizeof(T)); }
     
             const std::vector<int>& get_shape() const { return tensor_object->data.shape(); }
     
             const std::vector<int>& get_data() const { return tensor_object->data; }
     
             constexpr bool is_trainable() const { return _mutable; }
     
     ````        
     * ####Exception safety
     for

* Management class
    * This class is able to manage resources used in 
    _graph-construction process_
    
    contains: 
    * adjacency matrix for representing graph structure
    * mutex for modifying adj matrix
    * getters and setters for adj matrix
    * abilities to manage placeholders (which can stream data into graph)
    * declaration of private resources:
    
    ````cpp
    
            static std::deque<std::unique_ptr<Tensor_container<T>>> placeHolders;
    
            static std::deque<std::vector<std::unique_ptr<Tensor_container<T>>>> adj_forward;
    
            Management(){} ///disable the constructor
    
            static std::mutex adj_mutex;
    ````
    