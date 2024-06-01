# Heart-Attack-Prediction-Logistic-Regression-
# Heart Attack Risk Predictor

## Overview

This project provides a machine learning pipeline that predicts the risk of heart attack based on various health and lifestyle factors. The model is trained using a logistic regression algorithm and the pipeline is built using scikit-learn. The prediction is made through a user-friendly interface created with Gradio.

## Dataset

The dataset used for training the model is named `heart_attack_prediction_dataset.csv`. This dataset contains various features related to a person's health and lifestyle which are used to predict the risk of heart attack.

## Features

- **Patient ID:** Unique identifier for each patient.
- **Age:** Age of the patient.
- **Sex:** Gender of the patient.
- **Cholesterol:** Cholesterol level of the patient.
- **Blood Pressure:** Blood pressure of the patient.
- **Heart Rate:** Heart rate of the patient.
- **Diabetes:** Whether the patient has diabetes.
- **Family History:** Whether there is a family history of heart problems.
- **Smoking:** Whether the patient smokes.
- **Obesity:** Whether the patient is obese.
- **Alcohol Consumption:** Whether the patient consumes alcohol.
- **Exercise Hours Per Week:** Number of hours the patient exercises per week.
- **Diet:** Diet quality of the patient.
- **Previous Heart Problems:** Whether the patient had previous heart problems.
- **Medication Use:** Whether the patient uses any medication.
- **Stress Level:** Stress level of the patient.
- **Sedentary Hours Per Day:** Number of sedentary hours per day.
- **Income:** Income level of the patient.
- **BMI:** Body Mass Index of the patient.
- **Triglycerides:** Triglycerides level of the patient.
- **Physical Activity Days Per Week:** Number of days the patient is physically active per week.
- **Sleep Hours Per Day:** Number of sleep hours per day.
- **Country:** Country of residence.
- **Continent:** Continent of residence.
- **Hemisphere:** Hemisphere of residence.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/heart-attack-risk-predictor.git
    cd heart-attack-risk-predictor
    ```

2. Install the required packages:
    ```sh
    pip install pandas scikit-learn joblib gradio
    ```

3. Place the dataset (`heart_attack_prediction_dataset.csv`) in the project directory.

## Usage

1. Train the model and save the pipeline:
    ```python
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Load the dataset
    df = pd.read_csv('heart_attack_prediction_dataset.csv')

    # Separate target and features
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handling non-numeric columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Define the column transformer with an imputer for missing values
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Define the model
    model = LogisticRegression(max_iter=1000, solver='saga')

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Save the fitted pipeline
    joblib.dump(pipeline, 'trained_model_pipeline.joblib')
    ```

2. Run the Gradio interface:
    ```python
    import pandas as pd
    import joblib
    import gradio as gr

    # Load the trained model pipeline
    pipeline = joblib.load('trained_model_pipeline.joblib')

    # Define the prediction function
    def predict_heart_attack(
            patient_id, age, sex, cholesterol, blood_pressure, heart_rate, diabetes, 
            family_history, smoking, obesity, alcohol_consumption, exercise_hours, diet, 
            previous_heart_problems, medication_use, stress_level, sedentary_hours, income, 
            bmi, triglycerides, physical_activity, sleep_hours, country, continent, hemisphere
        ):
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'Patient ID': [patient_id],
            'Age': [age],
            'Sex': [sex],
            'Cholesterol': [cholesterol],
            'Blood Pressure': [blood_pressure],
            'Heart Rate': [heart_rate],
            'Diabetes': [diabetes],
            'Family History': [family_history],
            'Smoking': [smoking],
            'Obesity': [obesity],
            'Alcohol Consumption': [alcohol_consumption],
            'Exercise Hours Per Week': [exercise_hours],
            'Diet': [diet],
            'Previous Heart Problems': [previous_heart_problems],
            'Medication Use': [medication_use],
            'Stress Level': [stress_level],
            'Sedentary Hours Per Day': [sedentary_hours],
            'Income': [income],
            'BMI': [bmi],
            'Triglycerides': [triglycerides],
            'Physical Activity Days Per Week': [physical_activity],
            'Sleep Hours Per Day': [sleep_hours],
            'Country': [country],
            'Continent': [continent],
            'Hemisphere': [hemisphere]
        })

        # Make prediction
        prediction = pipeline.predict(input_data)
        
        # Return the prediction
        return "High Risk" if prediction == 1 else "Low Risk"

    # Create the Gradio interface
    interface = gr.Interface(
        fn=predict_heart_attack,
        inputs=[
            gr.Textbox(label="Patient ID"),
            gr.Number(label="Age"),
            gr.Radio(['Male', 'Female'], label="Sex"),
            gr.Number(label="Cholesterol"),
            gr.Textbox(label="Blood Pressure"),
            gr.Number(label="Heart Rate"),
            gr.Checkbox(label="Diabetes"),
            gr.Checkbox(label="Family History"),
            gr.Checkbox(label="Smoking"),
            gr.Checkbox(label="Obesity"),
            gr.Checkbox(label="Alcohol Consumption"),
            gr.Number(label="Exercise Hours Per Week"),
            gr.Dropdown(['Healthy', 'Average', 'Unhealthy'], label="Diet"),
            gr.Checkbox(label="Previous Heart Problems"),
            gr.Checkbox(label="Medication Use"),
            gr.Number(label="Stress Level"),
            gr.Number(label="Sedentary Hours Per Day"),
            gr.Number(label="Income"),
            gr.Number(label="BMI"),
            gr.Number(label="Triglycerides"),
            gr.Number(label="Physical Activity Days Per Week"),
            gr.Number(label="Sleep Hours Per Day"),
            gr.Textbox(label="Country"),
            gr.Textbox(label="Continent"),
            gr.Textbox(label="Hemisphere")
        ],
        outputs=gr.Textbox(label="Heart Attack Risk Prediction"),
        title="Heart Attack Risk Predictor",
        description="Enter the details to predict the risk of heart attack."
    )

    # Launch the Gradio interface
    interface.launch()
    ```

## Contributing

My Main issue was to make gradio interface give accurate results that is related to my trained model as it always gives me an error.

## License

This project is took the Dataset from Kaggle. 

## Acknowledgments

This project uses the following libraries:
- pandas
- scikit-learn
- joblib
- gradio
