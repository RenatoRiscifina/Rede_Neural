# codigo/treinar_modelo.py
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from rede_neural import NeuralNetwork, one_hot_encode

# Carrega o dataset MNIST
print("Carregando o dataset MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.astype(np.float32) / 255.0  # Normaliza os pixels
y = y.astype(np.int32)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converte os rótulos para one-hot encoding
num_classes = 10
y_train_enc = one_hot_encode(y_train, num_classes)
y_test_enc = one_hot_encode(y_test, num_classes)

# Define os parâmetros da rede
input_size = 784    # 28x28 pixels
hidden_size = 64
output_size = 10    # Dígitos de 0 a 9
modelo_path = "modelo_treinado.npz"

# Instancia a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)

# Verifica se existe um modelo salvo para carregá-lo
if os.path.exists(modelo_path):
    print("Carregando modelo previamente salvo...")
    nn.load_model(modelo_path)
else:
    print("Nenhum modelo salvo encontrado. Iniciando treinamento do zero.")
    initial_epochs = 10000
    loss_history = nn.train(X_train, y_train_enc, epochs=initial_epochs)
    nn.save_model(modelo_path)

# Aqui você pode continuar treinando o modelo (treinamento incremental)
additional_epochs = 1000  # Ajuste conforme necessário
print(f"Iniciando treinamento incremental por mais {additional_epochs} épocas...")
incremental_loss_history = nn.train(X_train, y_train_enc, epochs=additional_epochs)
nn.save_model(modelo_path)

# Avaliação no conjunto de teste
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")