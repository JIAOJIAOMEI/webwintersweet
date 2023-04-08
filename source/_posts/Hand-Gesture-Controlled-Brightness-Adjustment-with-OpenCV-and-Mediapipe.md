---
mathjax: true
title: Hand Gesture Controlled Brightness Adjustment
date: 2023-04-02 21:09:27
tags:
  - Hand gesture
  - Mediapipe
  - OpenCV
  - Computer vision
  - Image processing
  - Real-time tracking
---

This code implements a hand gesture-controlled brightness adjustment using OpenCV and Mediapipe libraries. The program uses the camera feed to detect the landmarks of the user's hands and track the movement of the index finger tips.

The program first initializes the hand detector using the Mediapipe library and opens the camera. It then sets the resolution of the camera and enters a while loop to continuously read the camera feed.

The program detects the landmarks of the user's hands and tracks the movement of the index finger tips. It draws circles on the index finger tips and a line between the index finger tips of the two hands. It calculates the distance between the two index finger tips and adjusts the brightness of the camera feed based on the distance. 

The program displays the brightness level and distance on the screen using text annotations. The loop continues until the user presses the 'q' key to exit the program.

The steps involved in the program are:

1. Import the necessary libraries such as OpenCV, Mediapipe, and math.
2. Initialize the hand detector using the Mediapipe library.
3. Open the camera feed and set the camera resolution.
4. Read the camera feed frame by frame and detect the hand landmarks using the hand detector.
5. Draw landmarks and circles on the index finger tip of each hand.
6. Calculate the distance between the index finger tips of both hands and adjust the brightness of the camera feed based on the distance.
7. Display the camera feed with the brightness value and the distance between the index finger tips of both hands.
8. Stop the program if the 'q' key is pressed, release the camera, and close all windows.

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 4/2/23 22:31

import cv2
import numpy as np
import mediapipe as mp
import math

# initialize hand detector
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

# open camera
cap = cv2.VideoCapture(0)

# set camera resolution
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

while True:
    # read camera feed
    success, img = cap.read()
    if not success:
        print("Unable to read camera feed")
        break
    if img is None:
        continue
    img = cv2.flip(img, 1)

    # detect the hands
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        # check if both hands are detected
        if len(results.multi_hand_landmarks) == 2:
            # get the landmarks of the hands
            lmList1 = results.multi_hand_landmarks[0].landmark
            lmList2 = results.multi_hand_landmarks[1].landmark

            # get the landmarks for the index fingers
            h, w, c = img.shape
            indexTip1 = (int(lmList1[8].x * w), int(lmList1[8].y * h))
            indexTip2 = (int(lmList2[8].x * w), int(lmList2[8].y * h))

            # draw circles on the index finger tips
            cv2.circle(img, indexTip1, 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, indexTip2, 15, (255, 0, 0), cv2.FILLED)

            # calculate the distance between the index finger tips of the two hands
            distance = math.sqrt(
                (indexTip2[0] - indexTip1[0]) ** 2 + (indexTip2[1] - indexTip1[1]) ** 2)

            # draw a line between the index finger tips
            cv2.line(img, indexTip1, indexTip2, (255, 0, 0), 3)

            # adjust the brightness of the camera feed based on the distance
            # brightness is from 0 to 100
            # distance is from 0 to 1000
            brightness = distance / 5
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

            # show the brightness on the screen
            cv2.putText(img, f"Brightness: {brightness:.2f}", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

            # draw the distance on the screen
            cv2.putText(img, f"Distance: {distance:.2f} pixels", (10, 70), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
        elif len(results.multi_hand_landmarks) == 1:
            # if only one hand is detected, show the message on the screen
            cv2.putText(img, "Please detect two hands", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
            # show the blue circle on the index finger tip
            lmList1 = results.multi_hand_landmarks[0].landmark
            h, w, c = img.shape
            indexTip1 = (int(lmList1[8].x * w), int(lmList1[8].y * h))
            cv2.circle(img, indexTip1, 15, (255, 0, 0), cv2.FILLED)
        else:
            brightness = 0
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    # show the camera feed
    cv2.imshow("Image", img)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
```

![brightness](Hand-Gesture-Controlled-Brightness-Adjustment-with-OpenCV-and-Mediapipe/brightness.gif)

The full video can be accessed at https://drive.google.com/file/d/1jz8ETwaZC0zIfCRmNHkV5fICosQPOWay/view?usp=sharing. Please have fun!😄

### Reference

1. YouTube. (2021, March 30). *Gesture volume control | OPENCV python | computer vision*. YouTube. Retrieved April 2, 2023, from https://www.youtube.com/watch?v=9iEPzbG-xLE&list=PLMoSUbG1Q_r8jFS04rot-3NzidnV54Z2q 
