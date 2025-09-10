import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Preprocessing Components ---
# This part runs only once when the app starts.
try:
    # Replace 'catboost_price_predictor.pkl' with the actual path to your saved model file
    model = joblib.load('final_catboost_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Error: Model file 'catboost_price_predictor.pkl' not found.")
    print("Please make sure your model file is in the same directory as this script.")

# --- 2. Define the Prediction Function ---
def predict_price(num_room, area, lat, lon, 
                  tip_predlozheniya, seriya, otopleniye, sostoyaniye,
                  material_doma, god_postroyki, tekushchiy_etazh,
                  vsego_etazhey):

    # Check if the model was loaded
    if model is None:
        return "Error: The model could not be loaded. Check your file path."

    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([{
        'num_room': num_room,
        'area': area,
        'lat': lat,
        'lon': lon,
        'Тип предложения': tip_predlozheniya,
        'Серия': seriya,
        'Отопление': otopleniye,
        'Состояние': sostoyaniye,
        'Материал_дома': material_doma,
        'Год_постройки': god_postroyki,
        'Текущий_этаж': tekushchiy_etazh,
        'Всего_этажей': vsego_etazhey
    }])

    # Reorder columns to match the training data
    # This step is crucial for pipeline consistency
    feature_order = ['num_room', 'area', 'address','lat', 'lon',
                     'Тип предложения', 'Серия', 'Отопление',
                     'Состояние','Материал_дома', 'Год_постройки', 
                     'Текущий_этаж','Всего_этажей']
    
    # We are missing the 'address' column, so we'll add a placeholder to avoid errors.
    # The CatBoost model can handle this as long as the categorical_features list is correct.
    # For a robust solution, you would need to process this user input.
    input_data['address'] = 'dummy_address' 
    
    # Ensure the order of columns matches what the model was trained on
    input_data = input_data.reindex(columns=feature_order)

    # Make the prediction
    prediction = model.predict(input_data)[0]

    return f"Predicted Price: ${prediction:,.2f}"

# --- 3. Set up the Gradio Interface ---
# Define the input components for the UI
inputs = [
    gr.Number(label="Number of Rooms"),
    gr.Number(label="Area (m²)"),
    gr.Number(label="Latitude"),
    gr.Number(label="Longitude"),
    gr.Textbox(label="Offer Type (e.g., от агента, от собственника)"),
    gr.Textbox(label="Building Series (e.g., элитка, 106 серия)"),
    gr.Textbox(label="Heating Type (e.g., на газе, центральное)"),
    gr.Textbox(label="Condition (e.g., евроремонт, без ремонта)"),
    gr.Textbox(label="Building Material (e.g., кирпичный, монолитный)"),
    gr.Number(label="Year of Construction"),
    gr.Number(label="Current Floor"),
    gr.Number(label="Total Floors")
]

# Set up the output component
output = gr.Textbox(label="Prediction Result")

# Create and launch the Gradio interface
gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="Bishkek Real Estate Price Predictor",
    description="Enter the features of an apartment to get a price prediction."
).launch()
