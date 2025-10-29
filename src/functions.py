import streamlit as st
import json

def show_functions(config: dict):
    for key, value in config.items():
        st.markdown(f"**{key}**")
        st.markdown(value['description'])

        i = 0
        for par in value['parameters']:
            st.markdown(f"- *{par}*: {value['help'][i]}")
            i += 1
        
        if 
        st.divider()
        

    
