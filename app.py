import streamlit as st
import pandas as pd

from utils import upload_and_return_df, infer_causality, create_graph
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title = "CausaMartix", page_icon="🧬")

st.markdown("<h1 style='text-align: center;'> 🧬 Causal Matrix 🧬</h1>", unsafe_allow_html=True)
st.divider()
st.markdown("""
Explore a fully deployed causal model, seamlessly accessible within this app!
Powered by:
- **MLflow’s Model-from-Code logging** for model management
- **Databricks Serving Endpoints** for scalable, real-time model access
- **Databricks Apps** for user friendly interface to explore causal relationships
Upload your data, generate causal insights, and interact with the model right here.
""")

st.divider()
df = upload_and_return_df()
if df is not None:
    
    if st.button("Infer Causality"):
        causal_matrix = infer_causality(df)
        causal_df=pd.DataFrame(causal_matrix, index=list(df.columns), columns=list(df.columns))

        fig =create_graph(pd.DataFrame(causal_matrix) ,list(df.columns))
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Causal Matrix", divider=True)
            st.write(causal_df)
        with col2:
            st.subheader("Causal Graph", divider=True)
            st.image(buf)
 
    else:
        st.write("Click the button to infer the causal model.")
else:
    st.write("Upload or drop a CSV file to display the DataFrame.")
    
