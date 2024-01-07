import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
from functions import *
import supabase

# Supabase credentials
SUPABASE_URL = "https://nwdaiodjblxyxhwgdruk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53ZGFpb2RqYmx4eXhod2dkcnVrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwNDI3MzYxOSwiZXhwIjoyMDE5ODQ5NjE5fQ.hg4b6Or9m902Z4kSSpltcbd7at81dL7FM4HTHqCkyI8"

# Initialize Supabase client
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# Streamlit layout
st.set_page_config(page_title='YOLOv5 | rice padi and rice weeviles', 
                   page_icon=':padi:', 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
st.title("rice flowering Detection")
st.subheader("Utilizing Ultralytics YOLOv5 Model")

# Upload image
uploaded_image = st.file_uploader("Upload an image and the model will predict if there is a padi.", 
                                  type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save to Local
    file_dir = "temp_dir/"
    file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type}
    img = Image.open(uploaded_image)
    clear_dir(file_dir)
    with open(file_dir + uploaded_image.name,"wb") as f: 
      f.write(uploaded_image.getbuffer())         
    st.success("File Uploaded")
    
    # Load Model
    rf = Roboflow(api_key="Y95ShsJtRcyJAdaeUnOX")
    project = rf.workspace("school-ermrh").project("padi-detection")
    model = project.version(14).model

    # Detect Objects
    with st.spinner('Inferring...'):
        detections = model.predict(file_dir + uploaded_image.name, confidence=10, overlap=29).json()
      
    

    image = Image.open(file_dir + uploaded_image.name)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    with st.container():
        for box in detections['predictions']:
            color = "#4892EA"
            x1 = int(box['x']) - int(box['width']) / 2
            x2 = int(box['x']) + int(box['width']) / 2
            y1 = int(box['y']) - int(box['height']) / 2
            y2 = int(box['y']) + int(box['height']) / 2

            draw.rectangle([
                x1, y1, x2, y2
            ], outline=color, width=5)

            if True:
                text = box['class']
                text_size = font.getsize(text)

                # set button size + 10px margins
                button_size = (text_size[0]+20, text_size[1]+20)
                button_img = Image.new('RGBA', button_size, color)
                # put text on button with 10px margins
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

                # put button on source image in position (0, 0)
                image.paste(button_img, (int(x1), int(y1)))
        
        st.subheader('Detections')
        st.image(image)
        for i in detections['predictions']:
          st.success(i['confidence'])
        if detections['predictions'][0]['confidence'] > 0.10:
          supabase_client.table('main').upsert([{'sound': '1.0'}]).execute()

    with st.container():
        st.code(
            '''
            @software{yolov5,
            title = {YOLOv5 by Ultralytics},
            author = {Glenn Jocher},
            year = {2020},
            version = {7.0},
            license = {AGPL-3.0},
            url = {https://github.com/ultralytics/yolov5},
            doi = {10.5281/zenodo.3908559},
            orcid = {0000-0001-5950-6979}
            }
            '''
        )
    
