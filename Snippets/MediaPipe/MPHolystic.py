from pickle import TRUE
import cv2
import mediapipe as mp

import imutils
import numpy as np
import argparse
import os

import tempfile
from typing import NamedTuple

from pythonosc import udp_client

from joblib import dump,load
from hand_detection_utils import *
from SVM import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic;
mp_hands = mp.solutions.hands

# argparse helps writing user-friendly commandline interfaces
parser = argparse.ArgumentParser()
# OSC server ip
parser.add_argument("--ip", default='127.0.0.1', help="The ip of the OSC server")
# OSC server port (check on SuperCollider)
parser.add_argument("--port", type=int, default=57120, help="The port the OSC server is listening on")

# Parse the arguments
args = parser.parse_args()

# Start the UDP Client
client = udp_client.SimpleUDPClient(args.ip, args.port)

if __name__ == "__main__":
      
  counter = 0

# Usage info
  print('USAGE:')
  print('	-Before training generate the images for the the two classes press "a" for class 1 and "b" for class 2:')
  print('		-Press "a" to save class A images')
  print('		-Press "b" to save class B images')
  print('	-Press "t" to start SVM training (if a model has already been saved, it will be loaded)')
  print('	-Press "s" to send OSC messages/packets to Touch Designer (must be pressed after training)')
  print('	-Press "q" to stop sound and "q" to stop image capture')
  
  # initialize weight for running average
  aWeight = 0.5

  num_frames_train = 0

  # initialize num of frames
  num_frames = 0

  # For webcam input:
  cap = cv2.VideoCapture(0)

  # Initialize variables
  TRAIN = False  # If True, images for the classes are generated
  SVM = False  # If True classification is performed
  START_SOUND = False  # If True OSC communication with SC is started

  with mp_holistic.Holistic(
      #model_complexity = 2,
      enable_segmentation = True,
      refine_face_landmarks = True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      success, image = cap.read()
    #  image = imutils.resize(image, width=700)
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # flip the frame so that it's not in the mirror view
      #image = cv2.flip(image, 1)

      # clone the frame
      clone = image.copy()

      # get high and width of the frame
      (heigth, width) = image.shape[:2]

      # convert the image to grayscale and blur it
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (7, 7), 0)


      # to get the background, keep looking till a threshold is reached
		  # so that our running average model gets calibrated
      #if num_frames < 30:
      #  run_avg(gray, aWeight)
      #else:
			
			  # segment the hand region
        #hand = segment(gray)
			

			  # check whether hand region is segmented
        #if hand is not None:
				  # if yes, unpack the thresholded image and
				  # segmented region
          #(thresholded, segmented) = hand

				  # draw the segmented region and display the frame
          #cv2.drawContours(clone, segmented, -1, (0, 0, 255))
				
				
				  # Center of the hand
				  #c_x, c_y = detect_palm_center(segmented)
          #radius = 5
				  #cv2.circle(thresholded, (c_x, c_y), radius, 0, 1)
				
          #cv2.imshow("Thresholded", thresholded)


      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.left_hand_landmarks,
          mp_holistic.HAND_CONNECTIONS)
          #landmark_drawing_spec=mp_drawing_styles
          #.get_default_hand_landmarks_style())  
      landmarksLHand = []
      if results.left_hand_landmarks:
        for id, lm in enumerate(results.left_hand_landmarks.landmark):
            h, w, c = image.shape 
            cx, cy = int(lm.x * w), int(lm.y * h) 
            cv2.putText(image, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
            landmarksLHand.append((cx,cy))    
      mp_drawing.draw_landmarks(
          image,
          results.right_hand_landmarks,
          mp_holistic.HAND_CONNECTIONS)
          #landmark_drawing_spec=mp_drawing_styles
          #.get_default_hand_landmarks_style()) 
      landmarksRHand = []
      if results.right_hand_landmarks:
        for id, lm in enumerate(results.right_hand_landmarks.landmark):
            h, w, c = image.shape 
            cx, cy = int(lm.x * w), int(lm.y * h) 
            cv2.putText(image, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
            landmarksRHand.append((cx,cy))    
      #mp_drawing.draw_landmarks(
      #    image,
      #    results.face_landmarks,
      #    mp_holistic.FACEMESH_CONTOURS,
      #    landmark_drawing_spec=None,
      #    connection_drawing_spec=mp_drawing_styles
      #    .get_default_face_mesh_contours_style())
      #mp_drawing.draw_landmarks(
      #    image,
      #   results.face_landmarks,
      #    mp_holistic.FACEMESH_TESSELATION,
      #   landmark_drawing_spec=None,
      #    connection_drawing_spec=mp_drawing_styles
      #    .get_default_face_mesh_tesselation_style())  
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style()) 
      landmarksPose = []
      if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
          h, w, c = image.shape 
          cx, cy = int(lm.x * w), int(lm.y * h) 
          cv2.putText(image, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
          landmarksPose.append((cx,cy))    
      #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)         
      # Flip the image horizontally for a selfie-view display.     

      reproduced_image = image - clone
      cv2.imshow('MediaPipe Holistic', cv2.flip(reproduced_image, 1))

      #image_segment = segment(image)

      #if image_segment is not None:
        #(thresholded, segmented) = image_segment


      # increment the number of frames
      num_frames += 1

      if cv2.waitKey(5) & 0xFF == 27:
        break

      if TRAIN:
    
			# Check if directory for current class exists
        if not os.path.isdir('images/class_'+class_name):
          os.makedirs('images/class_'+class_name)

        if num_frames_train < tot_frames:
				  # Change rectangle color to show that we are saving training images
          text = 'Generating ' + str(class_name) + ' images'
          cv2.putText(clone, text, (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				
				  # Save training images corresponding to the class
          cv2.imwrite('images/class_'+class_name+'/img_'+str(num_frames_train)+'.png', reproduced_image)

				  # keep track of how many images we are saving
          num_frames_train += 1

        else:

          print('Class '+class_name+' images generated')
          TRAIN = False

      if SVM:
			  # Convert image frame to numpy array
        reproduced_clone = reproduced_image.copy()
        reproduced_clone = imutils.resize(reproduced_clone, width=350)
        image_vector = np.array(reproduced_clone)
      #  image_vector = image_vector[1:2:-1]
        


			  # Use trained SVM to  predict image class
        class_test = model.predict(image_vector.reshape(1, -1))

        if class_test == 0:
				  # print('Class:  A value: ('+str(c_x)+','+str(c_y)+')')
          counter = counter + 1
          print(counter)
          text = 'Class: A'
          print(text) 
        else:
				  # print('Class: B value: ('+str(c_x)+','+str(c_y)+')')
          text = 'Class: B'
          counter = counter - 50
          print(counter)
          print(text) 

        cv2.putText(clone, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			  # Here we send the OSC message corresponding

      # observe the keypress by the user
      keypress = cv2.waitKey(1) & 0xFF

		
		  # if the user pressed "q", then stop looping
      if keypress == ord("q"):
        break

		  # Generate class A images
      if keypress == ord("a"):
        print('Generating the images for class A:')
        TRAIN = True
        num_frames_train = 0
        tot_frames = 250
        class_name = 'a'

		  # Generate class B images
      if keypress == ord("b"):
        print('Generating the images for class B:')
        TRAIN = True
        num_frames_train = 0
        tot_frames = 250
        class_name = 'b'

		  # Train and/or start SVM classification
      if keypress == ord('t'):
        SVM = True

        if not os.path.isfile('modelSVM.joblib'):
          print('I am training a SVM classification')
          model = train_svm()
        else:
          model = load('modelSVM.joblib')
          print('I am starting a SVM classification')

		  # Start OSC communication and sound
      if keypress == ord('s'): 

        client.send_message('/globe_control', [1, 2, 3])     
        #client.send_message('/globe_control', "MESSAGGIO FELICIO")   
        print("sending FELICIO")  

		  # Stop OSC communication and sound
      if keypress == ord('q'):
        print('I am stopping OSC communciation')
			  # Send OSC message to stop the synth
        client.send_message("/globe_control", ['stop'])

  cap.release()
  cv2.destroyAllWindows()