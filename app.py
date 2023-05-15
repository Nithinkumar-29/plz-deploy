from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns([1,2,1])
with col2:
	st.title("Alzheimer's Disease Detection")
st.divider()
col1,col2,col3 = st.columns([1,0.5,1])
with col2:
	logo = Image.open("Logo.jpg")
	st.image(logo)
col1,col2,col3 = st.columns([2,4,1])
with col2:
	st.markdown("##### Alzheimer's disease: A devastating neurological condition")
col1,col2,col3 = st.columns([1,3,1])
with col2:
	st.markdown("(Let's raise awareness, support research, and provide care for those affected. Together, we can make a difference)")
st.divider()
st.header("Stages of Alzheimer's Disease")
col3,col4,col5 = st.columns([1,4,1])
with col4:
	stages = Image.open("Stages.png")
	st.image(stages)
st.divider()
st.header("Upload MRI scan for Diagnosis")
img = st.file_uploader("")
op = {0:"Result: Mildly demented",1:"Result: Moderately demented",2:"Result: Non demented",3:"Result: Very mildly demented"}
sug = {0:"1. Maintain a routine.,2. Engage in stimulating activities.,3. Create a safe environment.,4. Stay socially connected.,5. Seek support from professionals.,6. Use memory aids.,7. Manage other medical conditions.",
1:"1. Ensure a safe environment.,2. Assist with daily activities.,3. Establish a routine.,4. Maintain social engagement.,5. Use memory aids.,6. Seek professional support.,7. Encourage suitable physical activities.,8. Explore therapeutic interventions.,9. Ensure proper nutrition and hydration.,10. Provide emotional support.",
2:"1. Engage in mentally stimulating activities.,2. Stay physically active.,3. Maintain a balanced diet.,4. Keep social connections strong.,5. Manage stress effectively.,6. Prioritize quality sleep.,7. Stay intellectually curious.,8. Limit alcohol consumption and avoid smoking.,9. Monitor and manage chronic health conditions.",
3:"1. Maintain routine and structure.,2. Engage in mental stimulation.,3. Use memory aids.,4. Ensure a safe environment.,5. Encourage social interaction.,6. Seek professional guidance.,7. Monitor and manage health conditions.,8. Adopt a healthy lifestyle.,9. Provide emotional support."}
if(img is not None):
	img = tf.keras.utils.load_img(img,target_size=(224,224,3))
	img1 = tf.keras.preprocessing.image.img_to_array(img)
	img1 = np.expand_dims(img,axis=0)
	model = tf.keras.models.load_model("model.keras",compile=False)
	model.compile(optimizer='adam',loss='categorical_crossentropy')
	pred = model.predict(img1)
	label = np.argmax(pred)
	col1,col2,col3 = st.columns([2,1,2])
	with col2:
		st.image(img)
		st.markdown("##### "+op[label])
	st.divider()
	st.header("Suggestions")
	col1,col2,col3 = st.columns([1,1,1])
	with col2:
		sugs = sug[label].split(',')
		for i in sugs:
			st.write(i)