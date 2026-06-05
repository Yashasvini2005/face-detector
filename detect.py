import cv2

# Load face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start webcam
cap = cv2.VideoCapture(0)

print("Controls:")
print("Press 'q' to quit")
print("Press 's' to save screenshot")
print("Press 'e' to toggle eye detection")
print("")

# Settings
show_eyes = True

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale (works better for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 8, minSize=(100, 100))
    
    # Count number of faces
    face_count = len(faces)
    
    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Add label "Face" above each face
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Detect eyes inside face region
        if show_eyes:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
    
    # Display face count on top-left corner
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display eye detection status
    eye_status = "Eyes: ON" if show_eyes else "Eyes: OFF"
    cv2.putText(frame, eye_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Show instructions at bottom
    cv2.putText(frame, "Press q: Quit | s: Screenshot | e: Toggle Eyes", (10, frame.shape[0]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show the result
    cv2.imshow('Face Detector - Press q to quit', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        filename = "screenshot.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")
    elif key == ord('e'):
        show_eyes = not show_eyes
        status = "ON" if show_eyes else "OFF"
        print(f"Eye detection toggled: {status}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Face detector closed.")