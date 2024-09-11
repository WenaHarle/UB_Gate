cmake_minimum_required(VERSION 3.10)
project(ocr_project)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set Tesseract include and library directories manually
set(Tesseract_INCLUDE_DIRS /usr/include/tesseract)
set(Tesseract_LIBRARIES /usr/lib/x86_64-linux-gnu/libtesseract.so)

# Set Leptonica include and library directories manually
set(Leptonica_INCLUDE_DIRS /usr/include/leptonica)
set(Leptonica_LIBRARIES /usr/lib/x86_64-linux-gnu/liblept.so)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include directories for OpenCV, Tesseract, and Leptonica
include_directories(${OpenCV_INCLUDE_DIRS} ${Tesseract_INCLUDE_DIRS} ${Leptonica_INCLUDE_DIRS})

# Specify the target executable and link libraries
add_executable(ocr_project main.cpp)
target_link_libraries(ocr_project ${OpenCV_LIBS} ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES})

