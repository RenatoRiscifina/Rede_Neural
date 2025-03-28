# Projeto Rede Neural - Previsão de Dígitos

Este projeto implementa uma rede neural para prever dígitos desenhados à mão, utilizando o dataset MNIST. Ele inclui uma interface interativa para desenhar dígitos e visualizar as previsões.

## Estrutura do Projeto
`requirements.txt` # Dependências do projeto codigo/ 
`interface_interativa.py` # Interface interativa com Streamlit 
`rede_neural.py` # Implementação da rede neural 
`treinar_modelo.py` # Script para treinar o modelo dados/ 
`modelo_treinado.npz` # Modelo treinado salvo documentacao/ 
`README.md` # Documentação do projeto

## Funcionalidades

- **Treinamento da Rede Neural**: O script `treinar_modelo.py` treina uma rede neural com o dataset MNIST.
- **Interface Interativa**: A aplicação `interface_interativa.py` permite desenhar dígitos e prever o número utilizando a rede neural.
- **Salvamento e Carregamento do Modelo**: O modelo treinado é salvo em `dados/modelo_treinado.npz` e pode ser carregado para uso posterior.

## Requisitos

Certifique-se de instalar as dependências listadas em `requirements.txt`:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/7)
```
Como Usar
1. Treinar o Modelo
Execute o script `treinar_modelo.py` para treinar a rede neural:

python `[treinar_modelo.py](http://_vscodecontentref_/8)`

2. Executar a Interface Interativa
Inicie a interface interativa com Streamlit:

streamlit run `[interface_interativa.py](http://_vscodecontentref_/9)`

Desenhe um dígito na tela e veja a previsão feita pela rede neural.

Detalhes Técnicos
Rede Neural: Implementada em `codigo/rede_neural.py`, com uma camada oculta de 64 neurônios e funções de ativação Sigmoid e Softmax.

Dataset: Utiliza o MNIST, carregado via fetch_openml da biblioteca scikit-learn.

Interface: Desenvolvida com Streamlit e streamlit-drawable-canvas para desenhar os dígitos.
Contribuição

Sinta-se à vontade para contribuir com melhorias para este projeto. Sugestões e pull requests são bem-vindos!

Licença
Este projeto é de uso livre para fins educacionais e experimentais.
