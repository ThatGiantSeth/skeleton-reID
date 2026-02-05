import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication
import asyncio
from qasync import QEventLoop, asyncSlot

from ui import MainWindow

GRAY_COLOR = (64, 64, 64)
LINE_THICKNESS = 3
JOINT_RADIUS = 4
CONFIDENCE_COLOR_THRESHOLD = 0.6
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)

# most of this code taken from a provided OpenNI/NiTE example file, modified to use the UI instead of the original OpenCV window
def draw_limb(img, ut, j1, j2, col):
    (x1, y1) = ut.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
    (x2, y2) = ut.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)

    if (0.4 < j1.positionConfidence and 0.4 < j2.positionConfidence):
        c = GRAY_COLOR if (min(j1.positionConfidence, j2.positionConfidence) < CONFIDENCE_COLOR_THRESHOLD) else col
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, LINE_THICKNESS)

        c = GRAY_COLOR if (j1.positionConfidence < CONFIDENCE_COLOR_THRESHOLD) else col
        cv2.circle(img, (int(x1), int(y1)), JOINT_RADIUS, c, -1)

        c = GRAY_COLOR if (j2.positionConfidence < CONFIDENCE_COLOR_THRESHOLD) else col
        cv2.circle(img, (int(x2), int(y2)), JOINT_RADIUS, c, -1)


def draw_skeleton(img, ut, user, col):
    for idx1, idx2 in [(nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK),
                       # upper body
                       (nite2.JointType.NITE_JOINT_NECK, nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_LEFT_SHOULDER, nite2.JointType.NITE_JOINT_TORSO),
                       (nite2.JointType.NITE_JOINT_TORSO, nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_RIGHT_SHOULDER, nite2.JointType.NITE_JOINT_NECK),
                       # left hand
                       (nite2.JointType.NITE_JOINT_LEFT_HAND, nite2.JointType.NITE_JOINT_LEFT_ELBOW),
                       (nite2.JointType.NITE_JOINT_LEFT_ELBOW, nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       # right hand
                       (nite2.JointType.NITE_JOINT_RIGHT_HAND, nite2.JointType.NITE_JOINT_RIGHT_ELBOW),
                       (nite2.JointType.NITE_JOINT_RIGHT_ELBOW, nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       # lower body
                       (nite2.JointType.NITE_JOINT_TORSO, nite2.JointType.NITE_JOINT_LEFT_HIP),
                       (nite2.JointType.NITE_JOINT_LEFT_HIP, nite2.JointType.NITE_JOINT_RIGHT_HIP),
                       (nite2.JointType.NITE_JOINT_RIGHT_HIP, nite2.JointType.NITE_JOINT_TORSO),
                       # left leg
                       (nite2.JointType.NITE_JOINT_LEFT_FOOT, nite2.JointType.NITE_JOINT_LEFT_KNEE),
                       (nite2.JointType.NITE_JOINT_LEFT_KNEE, nite2.JointType.NITE_JOINT_LEFT_HIP),
                       # right leg
                       (nite2.JointType.NITE_JOINT_RIGHT_FOOT, nite2.JointType.NITE_JOINT_RIGHT_KNEE),
                       (nite2.JointType.NITE_JOINT_RIGHT_KNEE, nite2.JointType.NITE_JOINT_RIGHT_HIP)]:
        draw_limb(img, ut, user.skeleton.joints[idx1], user.skeleton.joints[idx2], col)

# -------------------------------------------------------------
# main program from here
# -------------------------------------------------------------
    
def init_capture_device():

    openni2.initialize()
    nite2.initialize()
    return openni2.Device.open_any()


def close_capture_device():
    nite2.unload()
    openni2.unload()

class SkeletonGrabber(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    skeleton_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.dev = init_capture_device()
        self.img = None

        dev_name = self.dev.get_device_info().name.decode('UTF-8')
        print("Device Name: {}".format(dev_name))
        self.use_kinect = False
        if dev_name == 'Kinect':
            self.use_kinect = True
            print('using Kinect.')

        try:
            self.user_tracker = nite2.UserTracker(self.dev)
        except utils.NiteError:
            print("Unable to start the NiTE human tracker. Check "
                "the error messages in the console. Model data "
                "(s.dat, h.dat...) might be inaccessible.")
            sys.exit(-1)
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.start()
        
        (self.img_w, self.img_h) = CAPTURE_SIZE_KINECT if self.use_kinect else CAPTURE_SIZE_OTHERS
    
    def capture_skeleton(self):
        ut_frame = self.user_tracker.read_frame()

        color_frame = self.color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_uint8()
        self.img = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8,
                            buffer=color_frame_data)
        if self.use_kinect:
            self.img = self.img[0:self.img_h, 0:self.img_w]
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        if ut_frame.users:
            for user in ut_frame.users:
                if user.is_new():
                    print("new human id:{} detected.".format(user.id))
                    self.user_tracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                    draw_skeleton(self.img, self.user_tracker, user, (255, 0, 0))
                    ## this logic needs to be changed to actually emit the skeleton data, and i need to figure out how to make it into batches of 50 to match window size
                    self.skeleton_ready.emit(self.img)
        
        self.frame_ready.emit(self.img)
    
    def update_ui(self):
        ## need to somehow make an async thread to send the data to the Pi so that it doesn't block frame updates once the streaming connection is implemented
        self.ui.update_ui(self.img)
    
    @asyncSlot(np.ndarray)
    async def send_skeleton_data(self, img):
        print(f"Sending skeleton data...")

def main():
    # start the UI
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    ui = MainWindow()
    ui.show()
    
    skeleton_grabber = SkeletonGrabber(ui)
    skeleton_grabber.frame_ready.connect(ui.update_ui)
    skeleton_grabber.skeleton_ready.connect(skeleton_grabber.send_skeleton_data)
    
    
    # pretend theres server code here
    ip = "192.168.168.2"
    port = 5555
    print(f"Connected to pretend server at {ip}:{port}")
    ui.update_connection_info(ip, port)
    
    app.aboutToQuit.connect(close_capture_device)

    timer = QTimer()
    timer.timeout.connect(skeleton_grabber.capture_skeleton)
    timer.start(30)

    with loop:
        loop.run_forever()

if __name__ == '__main__':
    main()