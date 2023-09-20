import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta



# Initialize an empty DataFrame to store detected faces
detected_faces_df = pd.DataFrame(columns=["Name", "Confidence", "Date", "Time"])

# Initialize a dictionary to track when each face was last detected
face_last_detected = {}

# Load the encoded file from pickle
print("Loading Encoded File .......")
try:
    file = open("EncodeFile.p", "rb")
    encode_list_known_with_names = pickle.load(file)
    file.close()
    encode_list_known, name_list = encode_list_known_with_names
    print("Encoded File loaded successfully")
except FileNotFoundError:
    print("The file 'EncodeFile' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

video = cv2.VideoCapture(0)

while True:
    process_this_frame = True
    ret, frame = video.read()
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        cur_face = face_recognition.face_locations(rgb_small_frame)
        cur_face_encode = face_recognition.face_encodings(rgb_small_frame, cur_face)

        face_names = []
        detected_faces_data = []

        for face_encoding in cur_face_encode:
            matches = face_recognition.compare_faces(encode_list_known, face_encoding)
            name = "unknown"

            face_distances = face_recognition.face_distance(encode_list_known, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = name_list[best_match_index]

                # Calculate the confidence as a percentage based on face distance
                confidence = (1 - face_distances[best_match_index]) * 100

                # Get the current timestamp
                timestamp = datetime.now()
                date = timestamp.strftime("%Y-%m-%d")
                time = timestamp.strftime("%I:%M:%S %p")  # Include AM or PM in time

                # Check if this face was detected in the last 3 hours
                last_detected_time = face_last_detected.get(name)
                if last_detected_time and timestamp - last_detected_time < timedelta(hours=3):
                    message = f"Attendance already marked for {name}"
                    print(message)
                else:
                    # Update the last detected time for this face
                    face_last_detected[name] = timestamp

                    # Store the detected face details in a dictionary
                    data = {"Name": name, "Confidence": f"{confidence:.0f}%", "Date": date, "Time": time}
                    detected_faces_data.append(data)

            face_names.append(name)

        # Displaying annotation
        for (top, right, bottom, left), name in zip(cur_face, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), -1)
            
            if "Attendance already marked" not in name:
                cv2.putText(frame, name, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
                detected_faces_df = pd.concat([detected_faces_df, pd.DataFrame(detected_faces_data)], ignore_index=True)
            else:
                cv2.putText(frame, "Attendance already marked", (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    process_this_frame = not process_this_frame

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Get the current date in the format DD-MM-YYYY
current_date = datetime.now().strftime("%d-%m-%Y")

# Construct the file name with the current date
file_name = f"Attendence_Sheet_{current_date}.xlsx"

# Save the detected faces DataFrame to an Excel file
detected_faces_df.to_excel(file_name, index=False)
