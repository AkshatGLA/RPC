import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
from tensorflow.keras.preprocessing import image

# Load models
alz_model = load_model(r'C:\RPC\alz_classifier.h5')
tumor_model = load_model(r'C:\RPC\braintumor_new (1).h5')
pneumonia_model = pickle.load(open(r"C:\GEN AI-PPROJECT\RPC\pneumonia_model.sav", 'rb'))

############  Welcome Message  ############
import time
st.set_page_config(
    page_title="Intellihealth-MRI-Scanner",
    page_icon=r"C:\Users\AKSHAT SOMANI\OneDrive\Desktop\icon.jpg",
    layout="centered",  # Same as bot code
    initial_sidebar_state="auto",
)
# Define the CSS styles as a string
css = """
<style>
body {
    background-color: '#FF0000';
}

h2 {
    color: rgb(56, 173, 177);
    text-align: center;
}

.custom-paragraph {
    font-size: 18px;
    text-align: center;
    color: #333;
}

.custom-welcome{
    # background-color: #D3D3D3;
    padding: 10px;
    # border: 3px solid black;
    border-radius: 10px;
    margin-top: 20%;
    font-family: 'Comic Sans MS';
}


</style>
"""

# Inject the CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Create a placeholder for the welcome message
welcome_message = st.empty()

if 'msg_displayed' not in st.session_state:          
    # session_state :-  is a special feature in Streamlit that allows you to keep track of variables and their values 
    #                   that persist as users interact with the app, even as the app is reloaded or refreshed.
    st.session_state.msg_displayed = False
    # .msg_displayed :- this creates a variable in the session_state and manage its value.

# Display the customized welcome message using the CSS styles
if not st.session_state.msg_displayed:
    welcome_message.markdown(
        """
        <div class='custom-welcome'>
            
          <h2>IntelliHealth</h2>
               
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.msg_displayed = True


# Wait for 5 seconds
time.sleep(3)

# Clear the welcome message
welcome_message.empty()

####################################

# # Streamlit interface
logo_path=r"C:\Users\AKSHAT SOMANI\OneDrive\Desktop\logo.png"
st.sidebar.image(logo_path, use_column_width=True)
nav_option = st.sidebar.selectbox(
    "Select Operation:",
    ("Alzheimer's Detection", "Brain Tumor Detection", "Pneumonia Detection")
)

# Function to preprocess image for Alzheimer's model
def preprocess_image_for_alzheimer(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Tumor model
def preprocess_image_for_tumor(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Pneumonia model
def preprocess_image_for_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(120, 120))  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

result=""

def alzheimer():
    global result
    col1, col2 = st.columns([4,1.5])

    with col2:
        memory = Image.open(r'C:\GEN AI-PPROJECT\Frontend\m3.jpg')
        st.image(memory,width=250)

    with col1:
        st.title("Alzheimer's Classification")
        st.write("Upload an MRI image to check for Alzheimer's disease.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="alz_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_alzheimer(img)
        prediction = alz_model.predict(img_array)
        index = prediction.argmax()
        if index == 0:
            result="Sorry buddy you have Alzheimer's.Its like you have mild Alzheimer's disease."
            # st.write(result)
        elif index == 1:
            result="Sorry buddy you have Alzheimer's.Its like you have moderate Alzheimer's disease."
            # st.write(result)
        elif index == 2:
            result="Its like you don't have Alzheimer's disease."
            # st.write(result)
        elif index == 3:
            result="Sorry buddy you have Alzheimer's.Its like you have very mild Alzheimer's disease."
            # st.write(result)
    return st.success(result)

def tumor():
    global result

    col1, col2 = st.columns([5,1])

    with col2:
        brain_tumor = Image.open(r'C:\GEN AI-PPROJECT\Frontend\brain.jpg')
        st.image(brain_tumor,width=150)

    with col1:
        st.title("Brain Tumor Classification")
        st.write("Upload an MRI image to check if you have a brain tumor.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="tumor_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_tumor(img)
        prediction = tumor_model.predict(img_array)
        index = prediction.argmax()
        if index == 0:
            result="Sorry Buddy,You have a Tumor.Its you have Glioma Tumor."
            # st.write(result)
        elif index == 1:
            result="Sorry Buddy,You have a Tumor.Its like you have Meningioma Tumor."
            # st.write(result)
        elif index==2:
            result="You do not have a Tumor."
            # st.write(result)

        elif index == 3:
            result="Sorry Buddy,You have a Tumor.Its like you have Pituitary Tumor."
            # st.write(result)
    return st.success(result)    

def pneumonia():
    global result

    col1, col2 = st.columns([5, 1])  # Adjust the ratio of columns as needed

    with col2:
        lungs = Image.open(r'C:\GEN AI-PPROJECT\Frontend\lungs1.jpg')
        st.image(lungs, width=150)

    with col1:    
        st.title("Pneumonia Detection")
        st.write("Upload a chest X-ray image to detect Pneumonia.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"], key="pneumonia_uploader")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        temp_file_path = os.path.join("tempDir", uploaded_file.name)
        
        # Process the file (e.g., save it locally)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        preprocessed_image = preprocess_image_for_pneumonia(temp_file_path)

        # col1, col2, col3 = st.columns([3, 1, 3])

        # with col2:
        #     Proceed = st.button('Proceed')

        # if Proceed:
        pneu_prediction = pneumonia_model.predict(preprocessed_image)
        if pneu_prediction[0] > 0.75:
            result='It looks like you have pneumonia, which is an infection in your lungs.'
        else:
            result='I’m pleased to inform you that you do not have pneumonia.'
        return  st.success(result)
        
def ai():
    global result
    import os
    import streamlit as st
    from dotenv import load_dotenv
    import google.generativeai as gen_ai

    # Load environment variables
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Set up Google Gemini AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-1.5-flash')
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Initialize history for each option if not present
    if "chat_history_tumor" not in st.session_state:
        st.session_state.chat_history_tumor = []
    if "chat_history_pneumonia" not in st.session_state:
        st.session_state.chat_history_pneumonia = []
    if "chat_history_alzheimer" not in st.session_state:
        st.session_state.chat_history_alzheimer = []

    # Clear history for other navigation options and start new chat session
    def update_chat_session(nav_option):
        # Check the current navigation option and update history accordingly
        if nav_option == "Brain Tumor Detection":
            # Clear history for other options
            st.session_state.chat_history_pneumonia = []
            st.session_state.chat_history_alzheimer = []

            # Use existing history or start fresh
            st.session_state.chat_session = model.start_chat(history=st.session_state.chat_history_tumor)

        elif nav_option == "Pneumonia Detection":
            # Clear history for other options
            st.session_state.chat_history_tumor = []
            st.session_state.chat_history_alzheimer = []

            # Use existing history or start fresh
            st.session_state.chat_session = model.start_chat(history=st.session_state.chat_history_pneumonia)

        elif nav_option == "Alzheimer's Detection":
            # Clear history for other options
            st.session_state.chat_history_tumor = []
            st.session_state.chat_history_pneumonia = []

            # Use existing history or start fresh
            st.session_state.chat_session = model.start_chat(history=st.session_state.chat_history_alzheimer)

        # Save the current history for the selected option
        if nav_option == "Brain Tumor Detection":
            st.session_state.chat_history_tumor = st.session_state.chat_session.history
        elif nav_option == "Pneumonia Detection":
            st.session_state.chat_history_pneumonia = st.session_state.chat_session.history
        elif nav_option == "Alzheimer's Detection":
            st.session_state.chat_history_alzheimer = st.session_state.chat_session.history

    # Call the function with the selected navigation option
    update_chat_session(nav_option)
    # Initialize the question-answering chain in Streamlit session state if not already present
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = []
    if nav_option == "Brain Tumor Detection":
        st.session_state.qa_chain = []
    if nav_option=="Pneumonia Detection":
        st.session_state.qa_chain = []

    # Define function to translate roles for Streamlit
    def translate_role_for_streamlit(user_role):
        if user_role == "model":
            return "IntelliHealth"
        elif user_role == "user":
            return "Patient"
        else:
            return "IntelliHealth"  # Default to "assistant" if role is unknown

    # Function to create a prompt with context
    def create_prompt_with_context(user_prompt, context):
        return f"Context: {context}\n\nPatient: {user_prompt}\nIntelliHealth:"

    # Function to create the prompt based on the `result` content
    def get_prompt(result):
        if result:
            return f"""
You are a compassionate and knowledgeable health advisor who has been given the following information: {result}.

If a user asks any queries related to this information, provide detailed, easy-to-understand responses. Approach each query with empathy and positivity. Make sure your responses:

Address the user's questions thoroughly: Provide clear, actionable advice based on the data in {result}.
Offer encouragement and reassurance: Let the user know that they are in good hands, and nothing is too worrying. Use positive language to boost their confidence.
Be friendly and supportive: Use a conversational tone, and include light humor if appropriate to make the user feel at ease.
Guide the user: Offer practical advice and next steps to help them manage their health concerns.
Assure them of their well-being: Remind them that they will get better and encourage them to stay positive.
        """
        else:
            return (
                "You are a virtual doctor assistant. Your role is to provide medical advice "
                "based on the user’s symptoms or medical queries. Your responses should include "
                "symptom descriptions, precautions, advice, and encouragement to seek professional help."
            )

    # Display the chat history
    st.subheader("👋 Hey there! I'm IntelliHealth, your trusty healthcare Saathi! 🩺✨ Let's take care of your health together! 💪😊")
    for message in st.session_state.chat_session.history:
        if message.role != "context":  # Ensure that context is not displayed
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        # Determine the context prompt based on whether `result` has content
        context_prompt = get_prompt(result)

        # Create the prompt with context
        prompt_with_context = create_prompt_with_context(user_prompt, context_prompt)

        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message with context to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(prompt_with_context)

        # Add the question and answer to the QA chain
        st.session_state.qa_chain.append({"question": user_prompt, "answer": gemini_response.text})

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)



if nav_option=="Brain Tumor Detection":
    tumor()
    ai()

elif nav_option=="Alzheimer's Detection":   
    alzheimer()
    ai()
elif nav_option=="Pneumonia Detection":
    pneumonia()
    ai()







