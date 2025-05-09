import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Establecer ubicación de tracking local (para CI/CD y ejecución local)
mlflow.set_tracking_uri("file:./mlruns")

# Iniciar tracking
with mlflow.start_run():

    # 1. Registrar parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    # 2. Entrenamiento
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 3. Predicción y evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 4. Firma y logging del modelo
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        model,
        artifact_path="xgb_model",
        signature=signature,
        input_example=X_test[:5]
    )

    print("\n📦 Modelo registrado con MLflow.")

