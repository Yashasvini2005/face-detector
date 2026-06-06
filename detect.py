import cv2
import numpy as np
import os

# Load face detector
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load face cascade file")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Mode settings
# 0 = Rectangle mode (normal detection)
# 1 = Blur mode (privacy)
current_mode = 0

print("=" * 50)
print("FACE DETECTION SYSTEM")
print("=" * 50)
print("Controls:")
print("  Press 'q' - Quit")
print("  Press 's' - Save screenshot")
print("  Press 'm' - Switch modes")
print("")
print("Modes:")
print("  0: RECTANGLE MODE - Green box around faces")
print("  1: BLUR MODE - Faces blurred for privacy")
print("")
print("Starting with RECTANGLE MODE")
print("=" * 50)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    face_count = len(faces)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        if current_mode == 0:
            # MODE 0: Rectangle mode
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        elif current_mode == 1:
            # MODE 1: Blur mode
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            frame[y:y+h, x:x+w] = blurred
            cv2.putText(frame, "Private", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display current mode
    mode_names = ["RECTANGLE MODE", "BLUR MODE"]
    mode_colors = [(0, 255, 0), (0, 0, 255)]
    
    cv2.putText(frame, f"MODE: {mode_names[current_mode]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_colors[current_mode], 2)
    
    # Display face count
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Press 'm': Switch Modes | 's': Screenshot | 'q': Quit", 
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Show mode indicator
    if current_mode == 0:
        cv2.putText(frame, "[X] Rectangle     [ ] Blur", (frame.shape[1]-170, frame.shape[0]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "[ ] Rectangle     [X] Blur", (frame.shape[1]-170, frame.shape[0]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Show the result
    cv2.imshow('Face Detection System', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nQuitting...")
        break
        
    elif key == ord('s'):
        filename = f"screenshot_mode_{current_mode}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")
        
    elif key == ord('m'):
        current_mode = (current_mode + 1) % 2
        print(f"Mode changed to: {mode_names[current_mode]}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("Thanks for using Face Detection System!")
print(f"Final mode: {mode_names[current_mode]}")
print("=" * 50)