import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Importa a classe NeuralNetwork do nosso módulo da rede neural
from rede_neural import NeuralNetwork

# -------------------------------------------------------------------
# Configurações iniciais
# -------------------------------------------------------------------
st.title("PomboBot - Previsão de Dígitos")
st.subheader("Desenhe um dígito e veja a previsão da rede neural!")
st.write("Esta aplicação utiliza uma rede neural para prever dígitos desenhados à mão.")

# -------------------------------------------------------------------
# Configuração do Canvas
# -------------------------------------------------------------------
# Imagine o canvas como uma tela em branco em que você é o chef, criando sua própria obra!
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 255, 0.3)",  # Cor de preenchimento (não essencial)
    stroke_width=10,                    # Espessura da linha (como o traço do pincel)
    stroke_color="#FFFFFF",             # Cor do traço (branca para representar o dígito)
    background_color="#000000",         # Fundo preto, similar aos dados MNIST (fundo escuro)
    height=280,
    width=280,
    drawing_mode="freedraw",            # Permite desenhar livremente
    key="canvas",
)

# -------------------------------------------------------------------
# Instanciando a Rede Neural
# -------------------------------------------------------------------
# Assim como definimos uma receita, precisamos de um modelo para fazer a previsão.
# OBS: Para produção, recomenda-se carregar um modelo previamente treinado (salvo em arquivo).
input_size = 784    # 28x28 pixels
hidden_size = 64    # 64 neurônios na camada oculta
output_size = 10    # 10 classes para dígitos 0-9
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
# No exemplo, estamos instanciando o modelo. Se você tiver um modelo treinado,
# poderá carregá-lo (por exemplo, usando pickle) para evitar treinar do zero.

# -------------------------------------------------------------------
# Processamento dos Dados do Canvas e Previsão
# -------------------------------------------------------------------
if canvas_result.image_data is not None:
    # O canvas retorna uma imagem em formato RGBA. Primeiro, convertemos para uint8.
    img = canvas_result.image_data.astype(np.uint8)
    
    # Convertemos a imagem para escala de cinza.
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Redimensionamos a imagem para 28x28 pixels, o mesmo formato das imagens MNIST.
    img_resized = cv2.resize(img_gray, (28, 28))
    
    # Inverter as cores, se necessário (garante dígito branco em fundo preto).
    img_inverted = cv2.bitwise_not(img_resized)
    
    # Visualize a imagem processada (útil para entender como o modelo a "vê").
    st.image(img_inverted, caption="Imagem Processada - 28x28", width=140)
    
    # Prepara a imagem para o modelo: achatando e normalizando.
    X_input = img_inverted.flatten().reshape(1, -1).astype(np.float32)
    X_input /= 255.0  # Normalização: escalando os valores entre 0 e 1.
    
    # -------------------------------------------------------------------
    # Predição
    # -------------------------------------------------------------------
    # A rede neural faz a previsão com base no desenho, assim como você provaria o prato para saber se está bom.
    prediction = nn.predict(X_input)
    st.write(f"A rede neural prevê o dígito: **{int(prediction[0])}**")