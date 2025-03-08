execute_process(COMMAND "/home/kathan/catkin_ws/build/robotiq_1/robotiq_modbus_tcp/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/kathan/catkin_ws/build/robotiq_1/robotiq_modbus_tcp/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
