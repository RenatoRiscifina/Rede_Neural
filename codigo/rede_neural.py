# codigo/rede_neural.py
import numpy as np

def sigmoid(x):
    """
    Função de ativação sigmoide.
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivada da função sigmoide.
    """
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    """
    Função softmax para converter as saídas em probabilidades.
    """
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    """
    Converte os rótulos em codificação one-hot.
    """
    m = y.shape[0]
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), y] = 1
    return encoded

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Inicializa a rede neural com os tamanhos das camadas e taxa de aprendizado.
        """
        self.lr = learning_rate
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Realiza o forward propagation.
        """
        self.Z1 = np.dot(X, self.weights1) + self.bias1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights2) + self.bias2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def compute_loss(self, Y, output):
        """
        Calcula a função de perda usando cross-entropy.
        """
        m = Y.shape[0]
        log_likelihood = -np.log(output[range(m), np.argmax(Y, axis=1)] + 1e-9)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, Y, output):
        """
        Calcula os gradientes e atualiza os pesos da rede neural.
        """
        m = X.shape[0]
        dZ2 = output - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.weights2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.weights2 -= self.lr * dW2
        self.bias2 -= self.lr * db2
        self.weights1 -= self.lr * dW1
        self.bias1 -= self.lr * db1
    
    def train(self, X, Y, epochs=10):
        """
        Treina a rede neural por um número determinado de épocas.
        """
        loss_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(Y, output)
            loss_history.append(loss)
            self.backward(X, Y, output)
            print(f"Época {epoch+1}/{epochs} - Loss: {loss:.4f}")
        return loss_history
    
    def predict(self, X):
        """
        Realiza a predição retornando o índice da classe com maior probabilidade.
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def save_model(self, filepath):
        """
        Salva os parâmetros do modelo (pesos e vieses) num arquivo NPZ.
        """
        np.savez(filepath, 
                 weights1=self.weights1, 
                 bias1=self.bias1, 
                 weights2=self.weights2, 
                 bias2=self.bias2)
        print(f"Modelo salvo em '{filepath}'")
    
    def load_model(self, filepath):
        """
        Carrega os parâmetros do modelo a partir de um arquivo NPZ.
        """
        data = np.load(filepath)
        self.weights1 = data['weights1']
        self.bias1 = data['bias1']
        self.weights2 = data['weights2']
        self.bias2 = data['bias2']
        print(f"Modelo carregado de '{filepath}'")