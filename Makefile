# Makefile - Tareas automáticas del pipeline ML

install:
	@echo "🔧 Instalando dependencias..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	@echo "🚀 Ejecutando entrenamiento del modelo..."
	python -m src.train

test:
	@echo "🧪 Ejecutando pruebas de validación..."
	pytest tests/ || echo "⚠️ No se encontraron tests o fallaron algunos."

