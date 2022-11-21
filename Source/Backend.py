import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

import argparse
from pythonosc import udp_client

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

#################

# Argparse helps writing user-friendly commandline interfaces
parser = argparse.ArgumentParser() 
# OSC server ip
parser.add_argument("--ip", default='127.0.0.1', help="The ip of the OSC server")
# OSC server port (check on SuperCollider)
parser.add_argument("--port", type=int, default=57120, help="The port the OSC server is listening on")

# Initializing Ports
port1 = 57121
port2 = 57122
port3 = 57123
port4 = 57124
port5 = 57125
port6 = 57126


to_scde = 0
portSCDE = 7771



# Parse the arguments
#args = parser.parse_args()

args, unknown = parser.parse_known_args()

# Start the UDP Client
client = udp_client.SimpleUDPClient(args.ip, args.port) 

client1 = udp_client.SimpleUDPClient(args.ip, port1) 
client2 = udp_client.SimpleUDPClient(args.ip, port2) 
client3 = udp_client.SimpleUDPClient(args.ip, port3) 
client4 = udp_client.SimpleUDPClient(args.ip, port4) 
client5 = udp_client.SimpleUDPClient(args.ip, port5) 
client6 = udp_client.SimpleUDPClient(args.ip, port6)

client_scde = udp_client.SimpleUDPClient(args.ip, portSCDE)

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

def map(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, z1 = landmark1
    x2, y2, z2 = landmark2
    x3, y3, z3 = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def classifyPose(landmarks, output_image, count_in , to_scde, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


    control = 0
    if left_elbow_angle > 162  and left_elbow_angle < 218 and right_elbow_angle > 252 and right_elbow_angle < 308:
        if left_shoulder_angle > 10 and left_shoulder_angle < 50 and right_shoulder_angle > 103 and right_shoulder_angle < 158:
            label = 'Bottled Water' #SX          
            control = 2/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)   
            client1.send_message('/motion_detected', 1)   
            client2.send_message('/motion_detected', 0)   
            client3.send_message('/motion_detected', 0)   
            client4.send_message('/motion_detected', 0)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0)  

            client_scde.send_message('/soundscape_modifier', to_scde)
           
    if left_elbow_angle > 52  and left_elbow_angle < 98 and right_elbow_angle > 137 and right_elbow_angle < 203:
        if left_shoulder_angle > 92 and left_shoulder_angle < 148 and right_shoulder_angle > 5 and right_shoulder_angle < 53:
            label = 'Bottled Water' #DX
            control = 2/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)
            client1.send_message('/motion_detected', 1)   
            client2.send_message('/motion_detected', 0)   
            client3.send_message('/motion_detected', 0)   
            client4.send_message('/motion_detected', 0)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0) 

            client_scde.send_message('/soundscape_modifier', to_scde)

    if left_elbow_angle > 151  and left_elbow_angle < 209 and right_elbow_angle > 211 and right_elbow_angle < 307:
        if left_shoulder_angle > 1 and left_shoulder_angle < 35 and right_shoulder_angle > 1 and right_shoulder_angle < 58:
            label = 'Unplug' #SX       
            control = -3/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)
            client1.send_message('/motion_detected', 0)   
            client2.send_message('/motion_detected', 1)   
            client3.send_message('/motion_detected', 0)   
            client4.send_message('/motion_detected', 0)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0) 

            client_scde.send_message('/soundscape_modifier', to_scde)


    if left_elbow_angle > 36  and left_elbow_angle < 119 and right_elbow_angle > 149 and right_elbow_angle < 208:
        if left_shoulder_angle > 4 and left_shoulder_angle < 78 and right_shoulder_angle > 1 and right_shoulder_angle < 37:
            label = 'Unplug' #DX 
            control = -3/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)
            client1.send_message('/motion_detected', 0)   
            client2.send_message('/motion_detected', 1)   
            client3.send_message('/motion_detected', 0)   
            client4.send_message('/motion_detected', 0)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0)

            client_scde.send_message('/soundscape_modifier', to_scde)

    if left_elbow_angle > 8  and left_elbow_angle < 95 and right_elbow_angle > 260 and right_elbow_angle < 340:
        if left_shoulder_angle > 20 and left_shoulder_angle < 100 and right_shoulder_angle > 50 and right_shoulder_angle < 110: 
            label = 'Driving'   
            control = 4/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)
            client1.send_message('/motion_detected', 0)   
            client2.send_message('/motion_detected', 0)   
            client3.send_message('/motion_detected', 1)   
            client4.send_message('/motion_detected', 0)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0)

            client_scde.send_message('/soundscape_modifier', to_scde)
            

    if left_elbow_angle > 50  and left_elbow_angle < 96 and right_elbow_angle > 265 and right_elbow_angle < 325:
        if left_shoulder_angle > 120 and left_shoulder_angle < 150 and right_shoulder_angle > 120 and right_shoulder_angle < 150:
            label = 'Solid Shampoo' 
            control = -2/50
            count_in = count_in + control
            to_scde += map(control, -10, 20, -0.5, 1)
            count_in = np.clip(count_in, -10, 20)
            #to_scde = map(count_in, -10, 20, 0, 1)
            to_scde = count_in
            to_scde = np.clip(to_scde, -10, 20)
            client.send_message('/globe_control', count_in)
            client1.send_message('/motion_detected', 0)   
            client2.send_message('/motion_detected', 0)   
            client3.send_message('/motion_detected', 0)   
            client4.send_message('/motion_detected', 1)   
            client5.send_message('/motion_detected', 0)   
            client6.send_message('/motion_detected', 0)

            client_scde.send_message('/soundscape_modifier', to_scde)
        
    if label != 'Unknown Pose':
            
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    print('control = ', control)
    #print('control mapped= ', map(control, -10, 20, -0.5, 1))
    print('control_sent ', count_in)
    #print('to_scde mapped = ', map(to_scde, -10, 20, 0, 1))
    print('to_scde = ', to_scde)
    #print('to_scde clipped = ', np.clip(to_scde, 0,1))
    #print('to_scde mapped and clipped = ', np.clip(map(to_scde, -10, 20, 0, 1), 0,1))
    #count_in = count_in + control

    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    np.clip(count_in, -10, 20)
    np.clip(to_scde, 0, 1)
    count_out = count_in
    to_scde_out = to_scde
    

    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label, count_out, to_scde_out

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
count_in = 0
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        
        frame, _ , count_out, to_scde = classifyPose(landmarks, frame, count_in, to_scde, display=False)
        count_in = count_out
        
    
    # Display the frame.
    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
