import pandas as pd

def load_data():
    """
    Carga el dataset de diabetes desde una URL pública.
    Retorna: DataFrame de pandas
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df
