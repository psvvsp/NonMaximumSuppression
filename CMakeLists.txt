# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.8)

project ("NonMaximumSuppression")

# Включите подпроекты.
add_subdirectory ("nmslib")
#add_subdirectory ("nmstest")
