import sys
from openni import openni2, nite2, utils
import numpy as np
from mlsocket import MLSocket
import time

HOST = '192.168.137.2'
PORT = 55773
BATCH_SIZE = 100
    

openni2.initialize()
nite2.initialize()

dev = openni2.Device.open_any()

try:
    userTracker = nite2.UserTracker(dev)
except utils.NiteError as ne:
    logger.error("Unable to start the NiTE human tracker. Check "
                 "the error messages in the console. Model data "
                 "(s.dat, h.dat...) might be inaccessible.")
    sys.exit(-1)

while True:
    with MLSocket() as s:
        time.sleep(5)
        try:
            s.connect((HOST,PORT))
            print("Connected to receiver.")
            
            while True:
                try:   
                    skeleton_frames = []
                    n = 0
                    while n < BATCH_SIZE:
                        frame = userTracker.read_frame()
                
                        if frame.users:
                            for user in frame.users:
                                if user.is_new():
                                    print("New human detected! Calibrating...")
                                    userTracker.start_skeleton_tracking(user.id)
                                elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED:
                                    joints = user.skeleton.joints;
                                    
                                    frame_data = np.array([
                                    [joint.position.x, joint.position.y, joint.position.z]
                                    for joint in joints
                                    ])
                                    skeleton_frames.append(frame_data)
                                    n += 1
                    skeleton_array = np.array(skeleton_frames)
                    s.send(skeleton_array)
                    print(f"Sent batch of shape {skeleton_array.shape}")
                    
                    result = s.recv(1024)
                    print("ID:", int.from_bytes(result, 'big'))
                    
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    print("Connection was interrupted!")
                    break

        except (ConnectionRefusedError):
            print(f"The connection to {HOST} could not be completed. Retrying in 5 seconds.")
            
        except (KeyboardInterrupt):
            print("\nClient shutting down...")
            break
            
        
nite2.unload()
openni2.unload()