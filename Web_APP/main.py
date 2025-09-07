
import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Plant Identification","About"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT IDENTIFICATION SYSTEM")
    image_path = "home_page.gif"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Identification System! 🌿🔍
    
    Our mission is to help in plant identification efficiently by useing photos. After upload a image of a plant then  our system will analyze it to give a current health condition.
    ### How It Works
    1. **Upload Image:** Go to the **Plant Identification System** page and upload an image of a plant.
    2. **Analysis:** Our system will process the image using advanced algorithms to monitoring the current status.
    3. **Results:** After upload image click predict button to see the plant health condition.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art deep learning learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Plant Identification System** page in the sidebar to upload an image and experience the power of our Plant Health Monitoring System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


#Prediction Page
elif(app_mode=="Plant Identification"):
    st.header("Plant Identification")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Check")):
        st.snow()
        st.write("That is")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Black grass49_52', 'Charlock33_36', 'Cleavers25_28', 'Cranes_bill41_44', 'Fat Hen37_40', 'Fersken pileurt_dead_29', 'Field Pansy 45_47', 'Loose Silky_bent53_56', 'Majs1_4', 'Sugar beet9_12', 'Wheat5_8', 'chickweed17_20', 'scentless mayweed13_16', "shepherd's_purse 21_24"]
        st.success("That is: {}".format(class_name[result_index]))

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 404 rgb images of plant which is categorized into 14 different classes.
                A new directory containing 14 test images is created later for prediction purpose.
                #### Content
                1. train (309 images)
                2. test (14 images)
                3. validation (95 images)

                """)