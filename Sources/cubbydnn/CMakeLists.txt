# Target name
set(target CubbyDNN)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
)

# Sources
file(GLOB header_dir
        ${root_dir}/Includes/cubbydnn)

file(GLOB_RECURSE headers
        ${header_dir}/*.hpp)

file(GLOB_RECURSE sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target}
    ${sources} "Units/UnitType.cpp" "Engine/UnitManager.cpp")

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

option(WITH_BLAZE "Uses blaze for matrix computations" ON)

if(WITH_BLAZE)
    find_package(blaze)
    if( blaze_FOUND )

       add_library( blaze_target INTERFACE )
       target_link_libraries( blaze_target INTERFACE blaze::blaze )
       include_directories(BUILD_SHARED_LIBS)
       target_link_libraries(${target} PUBLIC blaze_target)

       add_definitions(-DWITH_BLAZE)
       message(STATUS "Using Blaze")
    endif()
endif()


# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)