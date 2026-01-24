import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
import time
import os

np.set_printoptions(threshold=sys.maxsize)

BATCH_SIZE = 1500
GRAY_COLOR = (64, 64, 64)
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)
OUTPUT_DIR = "./frames_default"


def parse_arg():
    parser = argparse.ArgumentParser(description='Test OpenNI2 and NiTE2.')
    parser.add_argument('-w', '--window_width', type=int, default=1024,
                        help='Specify the window width.')
    return parser.parse_args()

def draw_limb(img, ut, j1, j2, col):
    (x1, y1) = ut.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
    (x2, y2) = ut.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)

    if (0.4 < j1.positionConfidence and 0.4 < j2.positionConfidence):
        c = GRAY_COLOR if (j1.positionConfidence < 1.0 or j2.positionConfidence < 1.0) else col
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)

        c = GRAY_COLOR if (j1.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x1), int(y1)), 2, c, -1)

        c = GRAY_COLOR if (j2.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x2), int(y2)), 2, c, -1)


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

def init_capture_device():

    openni2.initialize()
    nite2.initialize()
    return openni2.Device.open_any()


def close_capture_device():
    nite2.unload()
    openni2.unload()
    

def capture_skeleton():
    args = parse_arg()
    dev = init_capture_device()
    
    dev_name = dev.get_device_info().name.decode('UTF-8')
    print("Device Name: {}".format(dev_name))
    use_kinect = False
    if dev_name == 'Kinect':
        use_kinect = True
        print('using Kinect.')
    try:
        userTracker = nite2.UserTracker(dev)
    except utils.NiteError as ne:
        logger.error("Unable to start the NiTE human tracker. Check "
                    "the error messages in the console. Model data "
                    "(s.dat, h.dat...) might be inaccessible.")
        sys.exit(-1)
        
    person = input("Please enter the person that you are generating training data for: ")
    type = input("Please enter the type of training data (walk, standing, etc.): ")
    OUTPUT_DIR = f"./frames_{person}{type}{BATCH_SIZE}"
    # Ensure output directory exists so cv2.imwrite won't fail
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Recording batch of {BATCH_SIZE} frames for \"{person}\"...")
    start_time = time.perf_counter()

    skeleton_frames = []
    n = 0
    
    (img_w, img_h) = CAPTURE_SIZE_KINECT if use_kinect else CAPTURE_SIZE_OTHERS
    win_w = args.window_width
    win_h = int(img_h * win_w / img_w)
    
    while n < (BATCH_SIZE - 1):
        frame = userTracker.read_frame()

        depth_frame = frame.get_depth_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        img = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                         buffer=depth_frame_data).astype(np.float32)
        if use_kinect:
            img = img[0:img_h, 0:img_w]

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(img)
        if (min_val < max_val):
            img = (img - min_val) / (max_val - min_val)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if frame.users:
            for user in frame.users:
                if user.is_new():
                    print("New person detected! Calibrating...")
                    userTracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                      
                    draw_skeleton(img, userTracker, user, (255, 0, 0))
                    joints = user.skeleton.joints;
                    
                    frame_data = np.array([
                    [joint.position.x, joint.position.y, joint.position.z]
                    for joint in joints
                    ])
                    skeleton_frames.append(frame_data)
                    n += 1

                # Convert floating depth image in range ~0..1 to uint8 0..255 for safe writing
                img_to_save = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                ok = cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{n:04d}.jpeg"), cv2.resize(img_to_save, (win_w, win_h)))
                if not ok:
                    print(f"Could not write image frame {n:04d} to directory {OUTPUT_DIR}!")
                    break

                cv2.imshow("Depth", cv2.resize(img, (win_w, win_h)))
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break    
                    
    skeleton_array = np.array(skeleton_frames)
    np.save(f"{person}{type}{BATCH_SIZE}.npy", skeleton_array)

    end_time = time.perf_counter()

    final_time = end_time - start_time
                
    # print(f"Successfully saved {BATCH_SIZE} frames for {person}. It took {final_time:.3f}s.\nLoading and printing array from file for verification:\n")

    # verify_array = np.load(f"{person}{type}{BATCH_SIZE}.npy")
    # print(f"Verification printout of shape {verify_array.shape}:\n{verify_array}")
            
    close_capture_device()

if __name__ == '__main__':
    capture_skeleton()