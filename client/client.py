import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication
import asyncio
from qasync import QEventLoop, asyncSlot
from preprocessing import normalize_skeleton
import struct
import time

from ui import MainWindow

LINE_THICKNESS = 3
JOINT_RADIUS = 4
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)

IP = "127.0.0.1"
PORT = 5555

# most of this code taken from a provided OpenNI/NiTE example file, modified to use the UI instead of the original OpenCV window
def joint_to_color_coords(ut, depth_stream, color_stream, joint):
    (dx, dy) = ut.convert_joint_coordinates_to_depth(joint.position.x, joint.position.y, joint.position.z)
    depth_x = int(round(dx))
    depth_y = int(round(dy))
    depth_z = int(round(joint.position.z))
    (cx, cy) = openni2.convert_depth_to_color(depth_stream, color_stream, depth_x, depth_y, depth_z)
    return (cx, cy)


def draw_limb(img, ut, depth_stream, color_stream, j1, j2):
    (x1, y1) = joint_to_color_coords(ut, depth_stream, color_stream, j1)
    (x2, y2) = joint_to_color_coords(ut, depth_stream, color_stream, j2)
    
    col = (255, 0, 0)

    if (0.4 < j1.positionConfidence and 0.4 < j2.positionConfidence):
        c = (64, 64, 64) if (min(j1.positionConfidence, j2.positionConfidence) < 0.6) else col
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, LINE_THICKNESS)

        c = (64, 64, 64) if (j1.positionConfidence < 0.6) else col
        cv2.circle(img, (int(x1), int(y1)), JOINT_RADIUS, c, -1)

        c = (64, 64, 64) if (j2.positionConfidence < 0.6) else col
        cv2.circle(img, (int(x2), int(y2)), JOINT_RADIUS, c, -1)


def draw_skeleton(img, ut, depth_stream, color_stream, user):
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
        draw_limb(img, ut, depth_stream, color_stream, user.skeleton.joints[idx1], user.skeleton.joints[idx2])

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
        self.skeleton_buffer = []
        self.buffer_size = 50

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
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.start()
        
        (self.img_w, self.img_h) = CAPTURE_SIZE_KINECT if self.use_kinect else CAPTURE_SIZE_OTHERS
    
    def capture_skeleton(self):
        ut_frame = self.user_tracker.read_frame()
        
        depth_frame = self.depth_stream.read_frame()
        color_frame = self.color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_uint8()
        self.img = np.frombuffer(color_frame_data, dtype=np.uint8).copy()
        self.img  = self.img.reshape((color_frame.height, color_frame.width, 3))
        
        if self.use_kinect:
            self.img = self.img[0:self.img_h, 0:self.img_w]
        
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        if ut_frame.users:
            for user in ut_frame.users:
                if user.is_new():
                    print("new human id:{} detected.".format(user.id))
                    self.user_tracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                    draw_skeleton(self.img, self.user_tracker, self.depth_stream, self.color_stream, user)
                    
                    # Buffer skeleton data for server
                    joints = user.skeleton.joints
                    skeleton_array = np.array([
                                    [joint.position.x, joint.position.y, joint.position.z]
                                    for joint in joints
                                    ])
                    self.skeleton_buffer.append(skeleton_array)
                    
                    # Only emit skeleton_ready when buffer is full
                    if len(self.skeleton_buffer) >= self.buffer_size:
                        batch = np.array(self.skeleton_buffer)
                        self.skeleton_ready.emit(batch)
                        self.skeleton_buffer = []
            
        ut_frame.close()
        color_frame.close()
        depth_frame.close()
        del ut_frame
        del color_frame
        del depth_frame
        self.frame_ready.emit(self.img)
    
    def update_ui(self):
        self.ui.update_ui(self.img)
        
class ServerHandler(QObject):
    connection_ready = pyqtSignal(str, int)
    connection_lost = pyqtSignal()
    result_ready = pyqtSignal(str, float, float)
    
    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.queue = asyncio.Queue()
        self.reader = None
        self.writer = None
        self.connected = False
        self.request_id = 0
        self.request_times = {}
        
    
    async def connect(self):
        while True:
            print(f"Looking for server on {self.ip}:{self.port}...")
            try:
                self.reader, self.writer = await asyncio.wait_for(asyncio.open_connection(self.ip, self.port), timeout = 5)
                print(f"Successfully connected to server at {self.ip}:{self.port}")
                self.connection_ready.emit(self.ip, self.port)
                self.connected = True
                await self.listen()
                self.writer.close()
            except (asyncio.TimeoutError,ConnectionRefusedError, OSError, ConnectionResetError):
                print(f"Unable to connect to server. Will retry in 5 seconds.")
                self.connected = False
                self.reader, self.writer = None, None
            await asyncio.sleep(5)
                
    async def listen(self):
        while True:
            try:
                data = await asyncio.wait_for(self.reader.readline(), timeout=3)
                result = data.decode(encoding='utf-8').strip()
                req_id, person_name, t = result.split(",")
                req_id = int(req_id)
                total_latency_ms = None
                start_time = self.request_times.pop(req_id, None)
                if start_time is not None:
                    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
                self.result_ready.emit(str(person_name), float(t), total_latency_ms)
                ## print(f"Debug output: id={req_id} person={person_name} t={t}")
            except asyncio.TimeoutError:
                continue
            
            if data == b"":
                # connection closed
                print("Connection closed by server.")
                self.connection_lost.emit()
                self.connected = False
                self.reader, self.writer = None, None
                break
          
    @asyncSlot(np.ndarray)
    async def send(self, batch):
        if self.connected and self.writer is not None:
            try:
                self.request_id += 1
                self.request_times[self.request_id] = time.perf_counter()
                batch = normalize_skeleton(batch)
                batch = np.ascontiguousarray(batch, dtype=np.float32)
                data = batch.tobytes()
                header = struct.pack('!I', self.request_id)

                self.writer.write(header + data)
                await self.writer.drain()
                print(f"Sent batch of size {batch.shape} to server - Request ID: {self.request_id}")
            except (ConnectionResetError, OSError):
                print("Connection lost while sending data.")
                self.writer, self.reader = None, None
                self.connected = False
                self.connection_lost.emit()
        

def main():
    
    # start the UI
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ui = MainWindow()
    ui.show()
    
    # connect to server
    # pretend theres server code here
    server = ServerHandler(IP, PORT)
    server.connection_ready.connect(ui.update_connection_info)
    loop.create_task(server.connect())
    server.connection_lost.connect(ui.update_connection_info)
    
    skeleton_grabber = SkeletonGrabber(ui)
    skeleton_grabber.frame_ready.connect(ui.update_ui)
    skeleton_grabber.skeleton_ready.connect(server.send)
    server.result_ready.connect(ui.update_results)
    
    app.aboutToQuit.connect(close_capture_device)

    timer = QTimer()
    timer.timeout.connect(skeleton_grabber.capture_skeleton)
    timer.start(30)

    with loop:
        loop.run_forever()

if __name__ == '__main__':
    main()