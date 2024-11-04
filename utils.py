import pandas as pd
import streamlit as st
from castle.algorithms import PC
import networkx as nx
import matplotlib.pyplot as plt
import os, requests

API_ROOT= os.getenv('API_ROOT')
API_TOKEN= os.getenv('API_TOKEN')
ENDPOINT_NAME= os.getenv('ENDPOINT_NAME')

def upload_and_return_df():
    # Allow only CSV files for upload
    uploaded_file = st.file_uploader("Upload or drop a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            return df  # Return the loaded DataFrame
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        return None


def infer_causality(df):

    data = {
            "inputs" : df.to_dict(orient='records'),
            }
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

    response = requests.post(
                                url=f"{API_ROOT}/serving-endpoints/{ENDPOINT_NAME}/invocations", json=data, headers=headers
                            )

    causal_matrix=response.json().get("predictions")
    return(causal_matrix)

    

def create_graph(causal_matrix, column_names):
    fig, _ = plt.subplots(figsize=(5, 5))  # Adjust the figsize to make the figure smaller
    fig.patch.set_alpha(0)
    g=nx.DiGraph(causal_matrix)
    nx.draw(
        G=g,
        node_color= "tab:blue",
        edge_color="y",
        node_size=1000,
        width=3,
        pos=nx.spring_layout(g, seed=123))
    pos=nx.spring_layout(g, seed=123)

    lables = {i: column_names[i] for i in range(len(column_names))}
    nx.draw_networkx_labels(g, pos, labels=lables, font_size=12, font_color="white");
    return(fig)


