import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import av


# load model
emotion_dict = {0:'angry', 1 :'fear', 2: 'happy', 3:'neutral', 4: 'sad'}
# load json and create model
json_file = open('model01.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("weights_emotions.hdf5")
size = (160, 160)

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self) -> None:
        self.min_confidence = 50

    def recv(self, frame):
        with mp_hands.Hands(
            model_selection=0,
            min_detection_confidence=(self.min_confidence / 100),
        ) as hands:
            img = frame.to_ndarray(format="bgr24")
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            # Draw the face detection annotations on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(1, 200, 7), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(2, 200, 2), thickness=2, circle_radius=2))
            cv2.imshow('Hand Tracking', img)                         
            return av.VideoFrame.from_ndarray(img, format="bgr24")


#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
    
        

def main():
    # Face Analysis Application #
    st.title("rAIn Application")
    st.subheader("Real Time Face Emotion and Hand Sign Detection")
    activiteis = ["Home", "Face Emotions Detection",'Hands Sign Detection','HSD', "Upload A Picture"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Mohammad Rachman """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                             this app uses OpenCV, Custom CNN model, mediapipe and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has three functionalities.

                 1. Real time face detection using web cam feed.
                 
                 2. Predict face emotion in picture.

                 3. Hand detection and tracking 

                 """)
    elif choice == "Face Emotions Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    
    elif choice == "Hands Sign Detection":

        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
        if ctx.video_processor:
            ctx.video_processor.min_confidence = st.slider(
                "Confidence[%]",
                min_value=0,
                max_value=100,
                step=1,
                value=50,
            )
    elif choice == 'HSD':
        st.header("Overview")
        st.write("""
         The ability to perceive the shape and motion of hands can be a 
         vital component in improving the user experience across a variety
         of technological domains and platforms. For example, it can form
         the basis for sign language understanding and hand gesture control, 
         and can also enable the overlay of digital content and information on 
         top of the physical world in augmented reality. While coming naturally to people, 
         robust real-time hand perception is a decidedly challenging computer vision task, 
         as hands often occlude themselves or each other (e.g. finger/palm occlusions and hand shakes) 
         and lack high contrast patterns
         
                 """)

        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
            while cap.isOpened():
                ret, frame = cap.read()
        
        # BGR 2 RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    
        # Set flag
                image.flags.writeable = False
        
        # Detections
                results = hands.process(image)
        
        # Set flag to true
                image.flags.writeable = True
        
        # RGB 2 BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
                print(results)
        
        # Rendering results
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(1, 200, 7), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(2, 200, 2), thickness=2, circle_radius=2),
                                         )
            
        
                cv2.imshow('Hand Tracking', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        
    elif choice == "Upload A Picture":
        st.write('please upload your picture down below')
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption='Uploaded file', use_column_width=True)
            image = ImageOps.fit(image,size)
            image = np.asarray(image)
            if st.button("Process"):
                face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                original_image = image.copy()
                faces = face_detector.detectMultiScale(original_image)
                roi = image[faces[0][1]:faces[0][1] + faces[0][2], faces[0][0]:faces[0][0] + faces[0][2]]
                roi = cv2.resize(roi, (48, 48))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = roi / 255
                roi = np.expand_dims(roi, axis = 0)
                probs = classifier.predict(roi)
                result = np.argmax(probs)
                if result == 0:
                    st.write("Its Angry Face")
                elif result == 1:
                    st.write("Its Fear Face")
                elif result == 2:
                    st.write('Its Happy Face')
                elif result == 3:
                    st.write('Its Neutral Face')
                elif result == 4:
                    st.write('Its Sad Face')
                else :
                    st.write('wrong picture')


    else:
        pass


if __name__ == "__main__":
    main()
