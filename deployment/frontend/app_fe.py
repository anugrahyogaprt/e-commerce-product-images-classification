import streamlit as st
import urllib.request
import numpy as np
import requests
import cv2

st.set_page_config(
    page_title='FTDS Phase 2 Milestone 2 -Anugrah Yoga Pratama',
    page_icon="✌️",
    initial_sidebar_state="collapsed",
    layout='centered',
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://github.com/anugrahyogaprt",
        'About': 'This is my Milestone 2\'s assignment, Enjoy :)'
    }
)

st.title("Fashion Image Classification")

col1, col2, col3 = st.columns([0.5, 5, 0.5])
col2.image('fashion.jpg', use_column_width=True, caption='Apparel and Footwear Fashion', width=500)
image_citation = '''
<p
        style="text-align: center;">
        Source : <a href:"https://www.fibre2fashion.com/news/apparel-news/apparel-footwear-groups-propose-europe-green-recovery-plan-268260-newsdetails.htm">Fibre2Fashion</a>
</p>
'''
st.markdown(image_citation, unsafe_allow_html=True)

st.write('Here we will try to predict the classification of fashion images with deep learning computers.\n\
         Enter data in the form of images. You can use the url of an image or you can upload your own image as input data.\n\
         The dataset used to train this learning comes from [Kaggle](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images).')

# ----------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
        st.subheader('Input')
        input_form = st.selectbox(
                label='Please select your input image:',
                options=['URL Link', 'Upload'],
                index=1
        )
# ----------------------------------------------------------------------
with col2:
        if input_form == "Upload":
                st.subheader('Upload')
                image_file = st.file_uploader('Choose a file', type=["png", "jpg","jpeg"])

                if image_file is not None:
                        # Convert the file to an opencv image (flatten np array)
                        img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(img_array, -1) #decode the data
                        # st.write(image_file)
                        # st.write(img_array)
                        # st.write(img.shape)

        else: 
                st.subheader('URL')
                image_url = st.text_input('Fill URL link in this form')
                if (len(image_url) > 0) & (('http' not in image_url) | ('://' not in image_url)):
                        st.write('Please type the URL correctly !')
                elif ('http' in image_url) & ('://' in image_url):
                        image_formats = ("image/png", "image/jpeg", "image/jpg")
                        r = requests.head(image_url)
                        if r.headers["content-type"] in image_formats:
                                url_response = urllib.request.urlopen(image_url)
                                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                                img = cv2.imdecode(img_array, -1)
                        else:
                                st.markdown('Please enter the image URL correctly !')
                                st.markdown('Image format : (JPG, JPEG, PNG)')

# ----------------------------------------------------------------------
def show_image(img_tensor):
        return col2.image(img_tensor, use_column_width=True, channels='BGR')

# ----------------------------------------------------------------------
# Buat fungsi untuk POST ke backend serta  menampilkan hasil
def result(img_array):
        data_inf = {'image_array': img_array.tolist()}
        # komunikasi
        URL = 'http://127.0.0.1:5000/fashion'
        r = requests.post(URL, json=data_inf)
        if col2.button('Predict Fashion Image'):
                prediction = r.json()['prediction']
                predict_proba = r.json()['predict_proba']
                col2.header(f'{prediction}')
                col2.subheader(f'True Positive Probability: {predict_proba*100:.2f}%')
        else:
                col2.write('Click to Predict')

# ----------------------------------------------------------------------
st.write("##")
st.write("##")

col1, col2, col3 = st.columns([1, 3, 1])
if input_form == "Upload":
        if image_file is not None:
                st.write("##")
                show_image(img)
                result(img)
else:
        if ('http' in image_url) & ('://' in image_url):
                if r.headers["content-type"] in image_formats:
                        st.write("##")
                        show_image(img)
                        result(img)


