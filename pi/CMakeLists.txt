PROJECT(pi-example)

SET(KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

ADD_DEFINITIONS("-DKERNEL_SOURCE_DIR=\"${KERNEL_SOURCE_DIR}\"")

ADD_EXECUTABLE(pi pi.cpp)
TARGET_LINK_LIBRARIES(pi HighResClock ${OpenCL_LIBRARIES})

