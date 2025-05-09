from src.data_loader import load_data

# 1. Carga los datos
df = load_data()
print("Primeras filas del dataset:")
print(df.head())

from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# 2. Verificar y manejar valores nulos
print("Valores nulos antes del tratamiento:")
print(df.isnull().sum())

# Si existieran nulos (no es el caso en este dataset), los llenaríamos:
df.fillna(df.median(numeric_only=True), inplace=True)

# 3. Separar variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 4. Escalado de las variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Verificar tamaños
print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")


# 6. Entrenamiento del modelo
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 7. Predicciones
y_pred = model.predict(X_test)

# 8. Evaluación del modelo
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# Iniciar una nueva corrida de MLflow
with mlflow.start_run():

    # 1. Registrar parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    # 2. Registrar métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 3. Firma del modelo (input/output)
    signature = infer_signature(X_test, y_pred)

    # 4. Guardar modelo entrenado
    mlflow.sklearn.log_model(
        model,
        artifact_path="xgb_model",
        signature=signature,
        input_example=X_test[:5]
    )

    print("\n📦 Modelo registrado con MLflow.")
