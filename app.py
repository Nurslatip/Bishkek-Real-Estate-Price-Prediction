import gradio as gr
import joblib
import pandas as pd

# --- 1. Load All Models ---
# This part runs only once when the app starts.
# We'll load all models to memory for quick access.
models = {}
try:
    models['CatBoost'] = joblib.load('final_catboost_model.pkl')
    models['RandomForest'] = joblib.load('rf_pipeline.pkl')
    models['LR'] = joblib.load('random_forest.pkl')
    print("All models loaded successfully.")
except FileNotFoundError:
    print("Error: One or more model files not found.")
    print("Please make sure all model files are in the same directory.")
    models = {} # Clear models if any are missing

# --- 2. Define the Prediction Function ---
def predict_price(selected_model, num_room, area, lat, lon,
                  tip_predlozheniya, seriya, otopleniye, sostoyaniye,
                  material_doma, god_postroyki, tekushchiy_etazh,
                  vsego_etazhey):
    
    # Check if the selected model exists
    if selected_model not in models or models[selected_model] is None:
        return "Error: The selected model could not be loaded."

    # Select the model based on user input
    model = models[selected_model]
    
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
    
    # This step is crucial for pipeline consistency, as before
    feature_order = ['num_room', 'area', 'address', 'lat', 'lon',
                     'Тип предложения', 'Серия', 'Отопление',
                     'Состояние', 'Материал_дома', 'Год_постройки',
                     'Текущий_этаж', 'Всего_этажей']
    
    input_data['address'] = 'dummy_address'
    input_data = input_data.reindex(columns=feature_order, fill_value=0) # Ensure all features are present

    # Make the prediction
    prediction = model.predict(input_data)[0]

    return f"Predicted Price: ${prediction:,.2f}"

# --- 3. Set up the Gradio Interface ---
# We add a Radio button for the user to choose the model
model_selector = gr.Radio(
    choices=['CatBoost', 'RandomForest', 'LR'],
    label="Choose a Model",
    value='CatBoost'  # Default value
)

inputs = [
    model_selector,  # This is the new input
    gr.Number(label="Number of Rooms"),
    gr.Number(label="Area (m²)"),
    gr.Number(label="Latitude"),
    gr.Number(label="Longitude"),
    gr.Textbox(label="Offer Type"),
    gr.Textbox(label="Building Series"),
    gr.Textbox(label="Heating Type"),
    gr.Textbox(label="Condition"),
    gr.Textbox(label="Building Material"),
    gr.Number(label="Year of Construction"),
    gr.Number(label="Current Floor"),
    gr.Number(label="Total Floors")
]

output = gr.Textbox(label="Prediction Result")

gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="Bishkek Real Estate Price Predictor",
    description="Enter the features of an apartment and choose a model to get a price prediction."
).launch()
