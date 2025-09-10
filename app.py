# app.py

import gradio as gr
from utils.predictor import get_prediction

# Define the Gradio interface
inputs = [
    gr.Number(label="Количество комнат"),
    gr.Number(label="Общая площадь (м²)"),
    gr.Number(label="Широта"),
    gr.Number(label="Долгота"),
    gr.Textbox(label="Серия дома"),
    gr.Textbox(label="Материал дома"),
    gr.Number(label="Всего этажей"),
    gr.Number(label="Текущий этаж"),
    gr.Textbox(label="Отопление"),
    gr.Textbox(label="Состояние")
]

outputs = [
    gr.Textbox(label="Прогноз CatBoost"),
    gr.Textbox(label="Прогноз Random Forest")
]

gr.Interface(
    fn=get_prediction,
    inputs=inputs,
    outputs=outputs,
    title="Bishkek Real Estate Price Prediction",
    description="This application predicts real estate prices in Bishkek using machine learning models.",
    css="""
        .main_row {
            padding: 10px;
        }
    """
).launch()
