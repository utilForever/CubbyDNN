# Target name
set(target Takion)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
)

#Added by Justin
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
)

# Sources
file(GLOB decl_dir
        ${root_dir}/Declarations/Takion)

file(GLOB impl_dir
        ${root_dir}/Implementations/Takion)

file(GLOB_RECURSE headers
        ${decl_dir}/*.hpp ${impl_dir}/*.hpp)

file(GLOB_RECURSE sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target}
    ${sources} )

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}

        INTERFACE
        )


# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)