import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email settings
EMAIL_ADDRESS = "aaryan.m299@ptuniv.edu.in"
EMAIL_PASSWORD = "pvlm nqrj ojkx rxvv"  # The provided app-specific password
RECIPIENT_EMAIL = "mohanaryan21@gmail.com"

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
conn = sqlite3.connect("criminal.db")
cursor = conn.cursor()

# Create a table for criminals
table_name = "criminals"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)

# Commit changes and close the connection
conn.commit()
conn.close()

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Criminal Recognition with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    # Insert data in database
    def record_criminal(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("criminal.db")
        cursor = conn.cursor()
        # Check if the criminal already has an entry for the current date
        cursor.execute("SELECT * FROM criminals WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            print(f"{name} is already marked as detected for {current_date}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO criminals (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            print(f"{name} marked as detected at {current_time} on {current_date}")
            self.send_email_notification(name, current_time, current_date)
        conn.close()

    # Send email notification
    def send_email_notification(self, name, time, date):
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = 'Criminal Detected'

            body = f"Criminal {name} detected at {time} on {date}."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, text)
            server.quit()

            print("Email sent successfully")

        except Exception as e:
            print(f"Failed to send email: {e}")

    def process(self, stream):
        while stream.isOpened():
            self.frame_cnt += 1
            flag, img_rd = stream.read()
            kk = cv2.waitKey(1)

            #  Press 'q' to quit
            if kk == ord('q'):
                break
            else:
                faces = detector(img_rd, 0)

                #  Update cnt for faces in frame
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # Update the centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list[:]
                self.current_frame_face_centroid_list = []

                # Update the name list
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.current_frame_face_name_list = []

                #  Update the features of faces in current frame
                self.current_frame_face_feature_list = []

                #  Position / 128D
                self.current_frame_face_position_list = []

                if len(faces) != 0:
                    #  Compute the face descriptors for faces in the current frame
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))

                        #  Get the positions of faces
                        self.current_frame_face_position_list.append(tuple(
                            [faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top()) / 4)]))

                        self.current_frame_face_centroid_list.append(
                            [int(faces[i].left() + faces[i].right()) / 2, int(faces[i].top() + faces[i].bottom()) / 2])

                    #  Re-classify the faces in the current frame
                    if self.frame_cnt == 1:
                        for i in range(len(faces)):
                            self.current_frame_face_name_list.append("unknown")

                    else:
                        self.current_frame_face_name_list = []
                        for k in range(len(faces)):
                            self.current_frame_face_name_list.append("unknown")

                        self.current_frame_face_X_e_distance_list = []

                        for i in range(len(self.current_frame_face_feature_list)):
                            current_frame_e_distance_list = []
                            for j in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[j][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[i],
                                                                                    self.face_features_known_list[j])
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    current_frame_e_distance_list.append(999999999)
                            self.current_frame_face_X_e_distance_list.append(current_frame_e_distance_list)
                        for i in range(len(self.current_frame_face_X_e_distance_list)):
                            min_distance = min(self.current_frame_face_X_e_distance_list[i])
                            if (min_distance < 0.4):
                                similar_person_num = self.current_frame_face_X_e_distance_list[i].index(min_distance)
                                self.current_frame_face_name_list[i] = self.face_name_known_list[similar_person_num]
                                self.record_criminal(self.current_frame_face_name_list[i])
                            else:
                                self.current_frame_face_name_list[i] = "unknown"
                self.draw_note(img_rd)
                self.update_fps()
                cv2.imshow("camera", img_rd)

        stream.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Face recognition with criminal detection system")
    fr = Face_Recognizer()
    if fr.get_face_database():
        stream = cv2.VideoCapture(0)
        fr.process(stream)
    else:
        logging.error("No face database found. Please ensure the feature database is created and accessible.")

if __name__ == '__main__':
    main()
