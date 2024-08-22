import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import os


st.set_page_config(layout='wide')

# Columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox('run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.empty()  # Use st.empty() to create a placeholder for dynamic text

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Configure the API with your key 
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the GenerativeModel
model = genai.GenerativeModel('gemini-1.5-flash')

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        finger = detector.fingersUp(hand)
        return lmList, finger
    else:
        return None

def draw(info, pre_pos, canvas):
    lmList, finger = info
    if finger == [0, 1, 0, 0, 0]:  # If index finger is up
        current_pos = lmList[8][0:2]
        if pre_pos is not None:
            cv2.line(canvas, pre_pos, current_pos, color=(255, 0, 255), thickness=10)
        pre_pos = current_pos
    elif finger == [1, 1, 0, 0, 0]:  # If all fingers are up (reset canvas)
        canvas = np.zeros_like(canvas)
        st.session_state["ai_response"] = ""  # Reset the AI response
    else:
        pre_pos = None
    return pre_pos, canvas

def callAi(pil_image):
    try:
        print("Calling the AI model...")
        response = model.generate_content(["Solve the math problem:", pil_image])
        print(f"Received response: {response.text}")
        st.session_state["ai_response"] = response.text  # Store the AI response in session state
        output_text_area.text(st.session_state["ai_response"])  # Update the UI with AI response
    except Exception as e:
        print(f"Failed to get a response from the API: {e}")
        st.session_state["ai_response"] = "Failed to get a response."
        output_text_area.text(st.session_state["ai_response"])  # Display the error message

pre_pos = None
canvas = None

# Initialize session state for AI response
if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""

while run:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        lmList, finger = info
        pre_pos, canvas = draw(info, pre_pos, canvas)
        if finger == [1, 1, 1, 1, 1] and st.session_state["ai_response"] == "":  # If index and middle fingers are up
            pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            callAi(pil_image)

    alpha = 0.7
    beta = 0.4
    gamma = 0

    image_combine = cv2.addWeighted(img, alpha, canvas, beta, gamma)

    FRAME_WINDOW.image(image_combine, channels='BGR')

    # Sleep to reduce CPU usage
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
