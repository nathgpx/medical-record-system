from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import uuid
from datetime import datetime

# Inicialização do Flask
app = Flask(__name__)

# Lista de doenças e sintomas
diseases = [
    "Gripe", "COVID-19", "Diabetes Tipo 2", "Hipertensão Arterial", "Asma",
    "Pneumonia", "Alergia Alimentar", "Doença de Alzheimer", "Depressão",
    "Anemia", "Câncer de Pulmão", "Doença de Crohn", "Fibromialgia",
    "Hipotireoidismo", "Esclerose Múltipla", "Síndrome do Intestino Irritável",
    "Artrite Reumatoide", "Doença Renal Crônica", "Enxaqueca", "Lúpus"
]

symptoms = [
    "Febre", "dor de garganta", "tosse", "dor no corpo", "tosse seca",
    "falta de ar", "perda de olfato", "dor de cabeça", "Sede excessiva",
    "fome frequente", "perda de peso", "visão embaçada", "fadiga",
    "Dor de cabeça", "tontura", "palpitações", "Urticária",
    "inchaço", "dor abdominal", "diarreia"
]

# Modelo de rede neural
encoder = OneHotEncoder()
labels_encoded = encoder.fit_transform(np.array(diseases).reshape(-1, 1)).toarray()

def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(len(symptoms),)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(diseases), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instancia o modelo
model = build_model()

# Dados de treino (simplificados)

        # Training data from original code
        self.training_data = np.array([
              # Gripe (e.g. presence of some symptoms)
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # COVID-19
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Diabetes Tipo 2
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    # Hipertensão Arterial
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    # Asma
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    # Pneumonia
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    # Alergia Alimentar
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # Doença de Alzheimer
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Depressão
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # Anemia
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    # Câncer de Pulmão
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    # Doença de Crohn
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    # Fibromialgia
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    # Hipotireoidismo
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    # Esclerose Múltipla
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    # Síndrome do Intestino Irritável
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    # Artrite Reumatoide
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    # Doença Renal Crônica
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    # Enxaqueca
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    # Lúpus
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        ])
])

# Treinar o modelo (simulação)
model.fit(training_data, labels_encoded, epochs=100, batch_size=4, verbose=0)

# Função para prever a doença
def predict_disease(symptom_vector):
    if len(symptom_vector) < len(symptoms):
        symptom_vector.extend([0] * (len(symptoms) - len(symptom_vector)))
    symptom_vector = np.array(symptom_vector).reshape(1, -1)
    prediction = model.predict(symptom_vector)
    disease_index = np.argmax(prediction)
    confidence = float(prediction[0][disease_index]) * 100
    return diseases[disease_index], confidence

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para diagnóstico
@app.route('/diagnose', methods=['POST'])
def diagnose():
    symptoms_input = request.form.getlist('symptoms')
    symptom_vector = [1 if symptom in symptoms_input else 0 for symptom in symptoms]
    disease, confidence = predict_disease(symptom_vector)
    return render_template('index.html', disease=disease, confidence=confidence, symptoms=symptoms_input)

if __name__ == '__main__':
    app.run(debug=True)
