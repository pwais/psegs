PSegs

readme!

 * trailer: 
     * show a histogram with examples of distance / orientation with samples over ALL datasets
     * show perf!  show time to fetch frames using Spark + Parquet
     * show a video of one camera with debug overlays.  maybe one with delauny lidar too (!)
     * show a frame HTML with 3d interface
     * show new things: argo associated bikes, delauny lidar, occlusion tree
 * supported datasets, how to get a blurb and **stats** on each of them.  prolly render histo reports for each.
 * data structures:
    * StampedDatum
    * Frame


cli
  * install as part of pip install psegs
  * dataset:
     * stages:
         - download (might be manual); download test fixtures
         - place (symlinks or whatever)
         - test (need way to check skipped tests)
         - demo (show one segment)
         - convert (all to sd table)





cd /tmp
pip3 install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg

rosdep init
rosdep update
mkdir ros_catkin_ws
cd ros_catkin_ws
catkin config --init -DCMAKE_BUILD_TYPE=Release -DROS_PYTHON_VERSION=3 --blacklist rqt_rviz rviz_plugin_tutorials librviz_tutorial --install

rosinstall_generator desktop_full --rosdistro melodic --deps --tar > melodic-desktop-full.rosinstall
wstool init -j8 src melodic-desktop-full.rosinstall

export ROS_PYTHON_VERSION=3
pip3 install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 wxPython


#!/bin/bash
#Check whether root
if [ $(whoami) != root ]; then
    echo You must be root or use sudo to install packages.
    return
fi

#Call apt-get for each package
for pkg in "$@"
do
    echo "Installing $pkg"
    sudo apt-get -my install $pkg >> install.log
done


chmod +x install_skip

#./install_skip `rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g" | sed -e "s/python/python3/g"`

echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata

apt-get install -y python3-psutil python3-catkin-pkg python3-empy python3-numpy python3-rospkg python3-yaml python3-pyqt5.qtwebkit python3-mock python3-rospkg python3-paramiko python3-cairo python3-pil python3-defusedxml python3-sip-dev python3-pyqt5.qtopengl python3-matplotlib python3-pyqt5 python3-pyqt5.qtsvg python3-sip-dev python3-pydot python3-pygraphviz python3-netifaces python3-yaml python3-opencv python3-catkin-pkg python3-rosdep python3-coverage python3-gnupg python3-lxml python3-mock python3-opengl python3-empy python3-nose

# apt-get install -y python3-wxtools

rosdep install --from-paths src --ignore-src -y --skip-keys="`rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g"`"
find . -type f -exec sed -i 's/\/usr\/bin\/env[ ]*python/\/usr\/bin\/env python3/g' {} +


cd src && git clone https://github.com/RobotWebTools/rosbridge_suite && git clone https://github.com/GT-RAIL/rosauth && cd -

# https://github.com/RobotWebTools/rosbridge_suite/blob/ad63eb1f7a05d8d52470ac1364b033c74683bbbf/rosbridge_server/package.xml#L18
apt-get install -y \
    python3-twisted python3-autobahn python-backports.ssl-match-hostname python3-tornado python3-bson

catkin build




#############################
# NOPE NOPE
cd /tmp
apt-get install -y lsb-release
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
apt-get update
apt-get install -y ros-melodic-rosbridge-server

roslaunch rosbridge_server rosbridge_websocket.launch
