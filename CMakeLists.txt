cmake_minimum_required(VERSION 3.25)
project(image-gp-2 VERSION 0.0.11)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/blt)
add_subdirectory(lib/blt-with-graphics)
add_subdirectory(lib/blt-gp)
add_subdirectory(lib/FastNoise2)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(lib/stb)

include_directories(include/)
include_directories(lib/implot)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB IMPLOT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/implot/*.cpp")

add_executable(image-gp-2 ${PROJECT_BUILD_FILES} ${IMPLOT_BUILD_FILES})

target_compile_options(image-gp-2 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
target_link_options(image-gp-2 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)

target_link_libraries(image-gp-2 PRIVATE BLT_WITH_GRAPHICS blt-gp ${OpenCV_LIBS})

if (${ENABLE_ADDRSAN} MATCHES ON)
    target_compile_options(image-gp-2 PRIVATE -fsanitize=address)
    target_link_options(image-gp-2 PRIVATE -fsanitize=address)
endif ()

if (${ENABLE_UBSAN} MATCHES ON)
    target_compile_options(image-gp-2 PRIVATE -fsanitize=undefined)
    target_link_options(image-gp-2 PRIVATE -fsanitize=undefined)
endif ()

if (${ENABLE_TSAN} MATCHES ON)
    target_compile_options(image-gp-2 PRIVATE -fsanitize=thread)
    target_link_options(image-gp-2 PRIVATE -fsanitize=thread)
endif ()
