import streamlit as st
import pickle
from PIL import Image



def image_sel(output):
    setosa = 'images/setosa.png'
    versicolor = 'images/versicolor.png'
    virginica = 'images/virginica.png'
    if output=='Iris-setosa':
        img = Image.open(setosa)
        st.image(img,caption='SETOSA')
    elif output=='Iris-versicolor':
        img = Image.open(versicolor)
        st.image(img,caption='VERSICOLOR')
    else:
        img = Image.open(virginica)
        st.image(img,caption='VIRGINICA')

st.title('ML Iris Classifier')
models = ['','SVM classifier', 'Decision Tree','KNN','Logistic Regression','Random Forest Classifier']
sel_model = st.selectbox('Choose Model',models)
sp_len=st.number_input('Enter Sepal Length')
sp_wid = st.number_input('Enter Sepal Width')
pt_len = st.number_input('Enter Petal Length')
pt_wid =st.number_input('Enter Petal Widht')
feature_data = [[sp_len,sp_wid,pt_len,pt_wid]]
predict_but = st.button('Predict')
if predict_but:
    if sel_model=='':
        st.write('Please Select Model')
    elif sel_model=='SVM classifier':
        with open('models/SVM.pickle','rb') as f:
            model1 = pickle.load(f)
        prediction = model1.predict(feature_data)
        image_sel(prediction[0])
    elif sel_model=='Decision Tree':
        with open('models/DTC.pickle', 'rb') as f2:
            model2 = pickle.load(f2)
        prediction2 = model2.predict(feature_data)
        image_sel(prediction2[0])
    elif sel_model=='KNN':
        with open('models/KNN.pickle', 'rb') as f3:
            model3 = pickle.load(f3)
        prediction3 = model3.predict(feature_data)
        image_sel(prediction3[0])
    elif sel_model=='Logistic Regression':
        with open('models/LOG.pickle', 'rb') as f4:
            model4 = pickle.load(f4)
        prediction4 = model4.predict(feature_data)
        image_sel(prediction4[0])
    else:
        with open('models/RFC.pickle','rb') as f5:
            model5 = pickle.load(f5)
        prediction5 = model5.predict(feature_data)
        image_sel(prediction5[0])


