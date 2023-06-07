import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import unicodedata

pickle_ = open('Sales.pkl','rb')
regressor = pickle.load(pickle_)

def welcome():
    return "Welcome All"

def predict_Bigmart_sales(Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type):
    # Normalize the Unicode inputs
    Item_Weight = unicodedata.normalize("NFKD", Item_Weight)
    Item_Fat_Content = unicodedata.normalize("NFKD", Item_Fat_Content)
    Item_Visibility = unicodedata.normalize("NFKD", Item_Visibility)
    Item_Type = unicodedata.normalize("NFKD", Item_Type)
    Item_MRP = unicodedata.normalize("NFKD", Item_MRP)
    Outlet_Establishment_Year = unicodedata.normalize("NFKD", Outlet_Establishment_Year)
    Outlet_Size = unicodedata.normalize("NFKD", Outlet_Size)
    Outlet_Location_Type = unicodedata.normalize("NFKD", Outlet_Location_Type)
    Outlet_Type = unicodedata.normalize("NFKD", Outlet_Type)
    # Perform the prediction
    prediction = regressor.predict([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type]])
    return prediction

def main():
    st.title("Big Mart Sales prediction ML App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Big Mart Sales Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Item_Weight = st.text_input("Item_Weight")
    Item_Fat_Content = st.text_input("Item_Fat_Content")
    Item_Visibility = st.text_input("Item_Visibility")
    Item_Type = st.text_input("Item_Type")
    Item_MRP = st.text_input("Item_MRP")
    Outlet_Establishment_Year = st.text_input("Outlet_Establishment_Year")
    Outlet_Size = st.text_input("Outlet_Size")
    Outlet_Location_Type = st.text_input("Outlet_Location_Type")
    Outlet_Type = st.text_input("Outlet_Type")
    result = ""
    if st.button("Predict Bigmart Sale"):
        result=predict_Bigmart_sales (Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year, 	Outlet_Size, Outlet_Location_Type, Outlet_Type)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("This is a Bigmart sales Prediction app which is used to predict Sales.")

if __name__=='__main__' :
    main()