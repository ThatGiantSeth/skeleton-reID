import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from ui import MainWindow

GRAY_COLOR = (64, 64, 64)
LINE_THICKNESS = 3
JOINT_RADIUS = 4
CONFIDENCE_COLOR_THRESHOLD = 0.6
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)

# most of this code taken from a provided OpenNI/NiTE example file, modified to use the UI instead of the original OpenCV window
def parse_arg():
    parser = argparse.ArgumentParser(description='Test OpenNI2 and NiTE2.')
    parser.add_argument('-w', '--window_width', type=int, default=1024,
                        help='Specify the window width.')
    return parser.parse_args()


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


def capture_skeleton():
    # start the UI
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    
    
    args = parse_arg()
    dev = init_capture_device()

    dev_name = dev.get_device_info().name.decode('UTF-8')
    print("Device Name: {}".format(dev_name))
    use_kinect = False
    if dev_name == 'Kinect':
        use_kinect = True
        print('using Kinect.')

    try:
        user_tracker = nite2.UserTracker(dev)
    except utils.NiteError:
        print("Unable to start the NiTE human tracker. Check "
              "the error messages in the console. Model data "
              "(s.dat, h.dat...) might be inaccessible.")
        sys.exit(-1)
    color_stream = dev.create_color_stream()
    color_stream.start()

    (img_w, img_h) = CAPTURE_SIZE_KINECT if use_kinect else CAPTURE_SIZE_OTHERS
    win_w = args.window_width
    win_h = int(img_h * win_w / img_w)

    def update_frame():
        ut_frame = user_tracker.read_frame()

        color_frame = color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_uint8()
        img = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8,
                         buffer=color_frame_data)
        if use_kinect:
            img = img[0:img_h, 0:img_w]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if ut_frame.users:
            for user in ut_frame.users:
                if user.is_new():
                    print("new human id:{} detected.".format(user.id))
                    user_tracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                    draw_skeleton(img, user_tracker, user, (255, 0, 0))
        
        ## may need to somehow make an async thread to send the data to the Pi so that it doesn't block frame updates once the streaming connection is implemented
        ui.update_ui(img)
            

    app.aboutToQuit.connect(close_capture_device)

    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(30)

    sys.exit(app.exec_())


if __name__ == '__main__':
    capture_skeleton()