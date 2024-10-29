import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe pose utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calcang(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    rad = math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0])
    ang = np.abs(rad * (180 / math.pi))
    ang = np.round(ang)
    
    if ang > 180.0:
        ang = 360.0 - ang

    print(ang)
    return ang

# Start video capture and pose detection
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose123:
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print("Camera Frame Empty")
            continue
        
        # Prepare the frame for pose detection
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        result = pose123.process(image)
        
        # Convert image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = result.pose_landmarks.landmark
            
            # Extract landmarks (positions of various body parts)
            nose = [landmarks[0].x, landmarks[0].y]
            l_shoulder = [landmarks[11].x, landmarks[11].y]
            r_shoulder = [landmarks[12].x, landmarks[12].y]
            l_elbow = [landmarks[13].x, landmarks[13].y]
            r_elbow = [landmarks[14].x, landmarks[14].y]
            l_wrist = [landmarks[15].x, landmarks[15].y]
            r_wrist = [landmarks[16].x, landmarks[16].y]
            l_hip = [landmarks[23].x, landmarks[23].y]
            r_hip = [landmarks[24].x, landmarks[24].y]
            l_knee = [landmarks[25].x, landmarks[25].y]
            r_knee = [landmarks[26].x, landmarks[26].y]
            l_ankle = [landmarks[27].x, landmarks[27].y]
            r_ankle = [landmarks[28].x, landmarks[28].y]
        
        except:
            pass
        
        # Draw the pose landmarks on the image
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Display the frame
        cv2.imshow('Posture Tracking Window', image)
        
        # Exit the loop when '0' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('0'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
