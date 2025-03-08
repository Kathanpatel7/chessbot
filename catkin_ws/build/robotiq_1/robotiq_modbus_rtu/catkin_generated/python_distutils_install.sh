#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/kathan/catkin_ws/src/robotiq_1/robotiq_modbus_rtu"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/kathan/catkin_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/kathan/catkin_ws/install/lib/python3/dist-packages:/home/kathan/catkin_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/kathan/catkin_ws/build" \
    "/usr/bin/python3" \
    "/home/kathan/catkin_ws/src/robotiq_1/robotiq_modbus_rtu/setup.py" \
     \
    build --build-base "/home/kathan/catkin_ws/build/robotiq_1/robotiq_modbus_rtu" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/kathan/catkin_ws/install" --install-scripts="/home/kathan/catkin_ws/install/bin"
