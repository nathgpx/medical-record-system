import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from datetime import datetime
import json
import uuid
import os
import random

class MedicalRecordSystem:
    def __init__(self):  # Corrected constructor
        # Lists from original code
        self.diseases = [
            "Gripe", "COVID-19", "Diabetes Tipo 2", "Hipertensão Arterial", "Asma",
            "Pneumonia", "Alergia Alimentar", "Doença de Alzheimer", "Depressão",
            "Anemia", "Câncer de Pulmão", "Doença de Crohn", "Fibromialgia",
            "Hipotireoidismo", "Esclerose Múltipla", "Síndrome do Intestino Irritável",
            "Artrite Reumatoide", "Doença Renal Crônica", "Enxaqueca", "Lúpus"
        ]

        self.symptoms = [
            "Febre", "dor de garganta", "tosse", "dor no corpo", "tosse seca",
            "falta de ar", "perda de olfato", "dor de cabeça", "Sede excessiva",
            "fome frequente", "perda de peso", "visão embaçada", "fadiga",
            "Dor de cabeça", "tontura", "palpitações", "Urticária",
            "inchaço", "dor abdominal", "diarreia"
        ]

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

        # Initialize the model
        self.encoder = OneHotEncoder()
        self.labels_encoded = self.encoder.fit_transform(np.array(self.diseases).reshape(-1, 1)).toarray()
        self.model = self._build_model()

        # Initialize records storage
        self.records_dir = "medical_records"
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)

    def _build_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(len(self.symptoms),)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.diseases), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        """Train the model with the predefined dataset"""
        self.model.fit(self.training_data, self.labels_encoded, epochs=100, batch_size=4, verbose=0)

    def symptoms_to_vector(self, symptom_list):
        """Convert list of symptoms to binary vector"""
        return [1 if symptom in symptom_list else 0 for symptom in self.symptoms]

    def predict_disease(self, symptom_vector):
        """Predict disease based on symptoms"""
        if len(symptom_vector) < len(self.symptoms):
            symptom_vector.extend([0] * (len(self.symptoms) - len(symptom_vector)))
        symptom_vector = np.array(symptom_vector).reshape(1, -1)
        prediction = self.model.predict(symptom_vector)
        disease_index = np.argmax(prediction)
        confidence = float(prediction[0][disease_index]) * 100
        return self.diseases[disease_index], confidence

    def create_medical_record(self, patient_data, symptoms, observations=""):
        """Create a new medical record"""
        symptom_vector = self.symptoms_to_vector(symptoms)
        predicted_disease, confidence = self.predict_disease(symptom_vector)

        record = {
            "record_id": str(uuid.uuid4()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient": {
                "id": patient_data.get("id", str(uuid.uuid4())),
                "name": patient_data.get("name", ""),
                "age": patient_data.get("age", ""),
                "gender": patient_data.get("gender", ""),
                "contact": patient_data.get("contact", "")
            },
            "consultation": {
                "symptoms": symptoms,
                "observations": observations,
                "predicted_disease": predicted_disease,
                "confidence": f"{confidence:.2f}%"
            },
            "status": "active"
        }

        # Save record to file
        filename = f"{self.records_dir}/{record['record_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)

        return record

    def get_medical_record(self, record_id):
        """Retrieve a medical record by ID"""
        try:
            with open(f"{self.records_dir}/{record_id}.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def update_medical_record(self, record_id, updates):
        """Update an existing medical record"""
        record = self.get_medical_record(record_id)
        if record:
            record.update(updates)
            with open(f"{self.records_dir}/{record_id}.json", 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=4)
            return record
        return None

    def get_patient_history(self, patient_id):
        """Retrieve all medical records for a specific patient"""
        patient_records = []
        for filename in os.listdir(self.records_dir):
            if filename.endswith('.json'):
                with open(f"{self.records_dir}/{filename}", 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    if record['patient']['id'] == patient_id:
                        patient_records.append(record)
        return patient_records

# Example usage
if __name__ == "__main__":  # Corrected main script check
    # Initialize the system
    mrs = MedicalRecordSystem()
    mrs.train_model()

    # Example: Create a new medical record
    patient_data = {
        "name": "João Silva",
        "age": 45,
        "gender": "M",
        "contact": "11999999999"
    }

    symptoms = ["Febre", "tosse", "falta de ar", "dor no corpo"]
    observations = "Paciente relata sintomas há 3 dias. Apresenta fadiga intensa."

    # Create record
    record = mrs.create_medical_record(patient_data, symptoms, observations)
    print("\nNovo Prontuário Médico criado:")
    print(json.dumps(record, ensure_ascii=False, indent=2))

    # Retrieve patient history
    patient_id = record["patient"]["id"]
    history = mrs.get_patient_history(patient_id)
    print(f"\nHistórico do paciente (total de {len(history)} registros):")
    for record in history:
        print(f"Data: {record['date']}")
        print(f"Doença prevista: {record['consultation']['predicted_disease']}")
        print(f"Confiança: {record['consultation']['confidence']}")
        print("---")




class ChatbotForm:
    def __init__(self, system):
        self.system = system

    def start_chat(self):
        print("Olá! Bem-vindo ao sistema de prontuários médicos.")
        patient_data = self.collect_patient_data()
        symptoms = self.collect_symptoms()
        observations = self.collect_observations()

        # Criar o prontuário médico
        record = self.system.create_medical_record(patient_data, symptoms, observations)
        print("\nProntuário Médico criado com sucesso!")
        print(record)

    def collect_patient_data(self):
        patient_data = {}
        patient_data["name"] = input("Por favor, digite o nome do paciente: ")
        patient_data["age"] = input("Digite a idade do paciente: ")
        patient_data["gender"] = input("Digite o gênero do paciente (M/F): ")
        patient_data["contact"] = input("Digite o contato do paciente: ")
        return patient_data

    def collect_symptoms(self):
        print("Quais sintomas o paciente está apresentando? (Digite 'fim' para encerrar)")
        symptoms = []
        while True:
            symptom = input("Sintoma: ")
            if symptom.lower() == 'fim':
                break
            symptoms.append(symptom)
        return symptoms

    def collect_observations(self):
        return input("Adicione observações sobre o paciente: ")

# Exemplo de uso
if __name__ == "__main__":
    mrs = MedicalRecordSystem()
    mrs.train_model()

    chatbot = ChatbotForm(mrs)
    chatbot.start_chat()
