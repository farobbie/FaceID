cmake_minimum_required(VERSION 3.0)

project(faceid)

Set(BINARY_NAME faceid)

set(CMAKE_AUTOUIC ON)
set (CMAKE_CXX_STANDARD 14)

aux_source_directory(./src SRC_LIST)



set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)

#add_subdirectory(./lib/dlib-19.13 dlib_build)
find_package( dlib REQUIRED )
find_package( OpenCV REQUIRED )

find_package(Qt5 REQUIRED COMPONENTS Widgets Core Network Gui)
qt5_wrap_ui(UI_HEADERS ./src/face_mainwindow.ui)
qt5_wrap_cpp(CPP_HEADERS ./src/face_mainwindow.h ./src/face_manager.h ./src/face_dlib.h ./src/face_worker.h)


add_executable(${BINARY_NAME} ${SRC_LIST} ${UI_HEADERS} ${CPP_HEADERS})


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
)

target_link_libraries("faceid"
dlib::dlib
${OpenCV_LIBS}
${Boost_LIBRARIES}
Qt5::Core
Qt5::Gui
Qt5::Widgets
)

INSTALL(TARGETS ${BINARY_NAME} DESTINATION bin)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -DUSE_AVX_INSTRUCTIONS=1")

if (CUDA_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDLIB_USE_CUDA")
else()
endif()
