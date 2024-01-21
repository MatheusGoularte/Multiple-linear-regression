import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Dados - Substitua com os seus próprios dados
data = {
    'variavel_1': [val1, val2, val3, val4, val5],
    'variavel_2': [val1, val2, val3, val4, val5],
    'variavel_3': [val1, val2, val3, val4, val5],
    'variavel_4': [val1, val2, val3, val4, val5],
    'variavel_dependente': [val1, val2, val3, val4, val5],
}

# Criar DataFrame
df = pd.DataFrame(data)

# Substitua com os nomes das suas variáveis
variaveis_independentes = ['variavel_1', 'variavel_2', 'variavel_3', 'variavel_4']
variavel_dependente = 'variavel_dependente'

# Função para criar DataFrame
def criar_dataframe(data):
    return pd.DataFrame(data)

# Função para separar variáveis
def separar_variaveis(df, variaveis_independentes, variavel_dependente):
    X = df[variaveis_independentes]
    Y = df[variavel_dependente]
    return X, Y

# Função para ajustar o modelo
def ajustar_modelo(X, Y):
    model = sm.OLS(Y, sm.add_constant(X)).fit()  # Adiciona a constante ao modelo
    return model

# Função para imprimir coeficientes
def imprimir_coeficientes(model, variaveis_independentes):
    print("\nCoeficientes Estimados: Peso de cada variável no sistema")
    print(f'Neste caso: { " + ".join([f"{coef:.4f}*{var}" for coef, var in zip(model.params[1:], variaveis_independentes)]) }')
    print(model.params)

# Função para plotar gráficos de dispersão
def scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(x, y)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Função para avaliar a equação
def avaliar_equacao(coeficientes, valores):
    return sum(c * v for c, v in zip(coeficientes, valores))

# Separar variáveis
X, Y = separar_variaveis(df, variaveis_independentes, variavel_dependente)

# Ajustar modelo
model = ajustar_modelo(X, Y)

# Imprimir coeficientes
imprimir_coeficientes(model, variaveis_independentes)

# Plotar gráficos de dispersão
for coluna in X.columns:
    scatter_plot(df[coluna], Y, coluna, variavel_dependente)

# Avaliar expressão para variação de variáveis
valores_variavel_1 = np.linspace(min(df['variavel_1']), max(df['variavel_1']), 20)
resultados_variavel_1 = [avaliar_equacao(model.params[1:], [v1, v2, v3, v4]) for v1 in valores_variavel_1]
plotar_resultados(valores_variavel_1, resultados_variavel_1, 'variavel_1', 'Resultado')

# Repita o processo para outras variáveis, se necessário
# valores_variavel_2 = np.linspace(min(df['variavel_2']), max(df['variavel_2']), 20)
# resultados_variavel_2 = [avaliar_equacao(model.params[1:], [v1, v2, v3, v4]) for v2 in valores_variavel_2]
# plotar_resultados(valores_variavel_2, resultados_variavel_2, 'variavel_2', 'Resultado')

# E assim por diante...