import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Data - Replace with your own data
data = {
    'variavel_1': [val1, val2, val3, val4, val5],
    'variavel_2': [val1, val2, val3, val4, val5],
    'variavel_3': [val1, val2, val3, val4, val5],
    'variavel_4': [val1, val2, val3, val4, val5],
    'variavel_dependente': [val1, val2, val3, val4, val5],
}

# Create DataFrame
df = pd.DataFrame(data)

# Replace with the names of your variables
variaveis_independentes = ['variavel_1', 'variavel_2', 'variavel_3', 'variavel_4']
variavel_dependente = 'variavel_dependente'

# Function to create DataFrame
def criar_dataframe(data):
    return pd.DataFrame(data)

# Function to separate variables
def separar_variaveis(df, variaveis_independentes, variavel_dependente):
    X = df[variaveis_independentes]
    Y = df[variavel_dependente]
    return X, Y

# Function to adjust the model
def ajustar_modelo(X, Y):
    model = sm.OLS(Y, sm.add_constant(X)).fit()  # Adds the constant to the model
    return model

# Function to print coefficients
def imprimir_coeficientes(model, variaveis_independentes):
    print("\nCoeficientes Estimados: Peso de cada variável no sistema")
    print(f'Neste caso: { " + ".join([f"{coef:.4f}*{var}" for coef, var in zip(model.params[1:], variaveis_independentes)]) }')
    print(model.params)

# Function for plotting scatter plots
def scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(x, y)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Function to evaluate the equation
def avaliar_equacao(coeficientes, valores):
    return sum(c * v for c, v in zip(coeficientes, valores))

# Separate variables
X, Y = separar_variaveis(df, variaveis_independentes, variavel_dependente)

# Adjust model
model = ajustar_modelo(X, Y)

# Print coefficients
imprimir_coeficientes(model, variaveis_independentes)

# Plotting scatter plots
for coluna in X.columns:
    scatter_plot(df[coluna], Y, coluna, variavel_dependente)

# Evaluate expression for variable variation
valores_variavel_1 = np.linspace(min(df['variavel_1']), max(df['variavel_1']), 20)
resultados_variavel_1 = [avaliar_equacao(model.params[1:], [v1, v2, v3, v4]) for v1 in valores_variavel_1]
plotar_resultados(valores_variavel_1, resultados_variavel_1, 'variavel_1', 'Resultado')

# Repeat the process for other variables, if necessary
# variable_values_2 = np.linspace(min(df['variable_2']), max(df['variable_2']), 20)
# results_variable_2 = [evaluate_equation(model.params[1:], [v1, v2, v3, v4]) for v2 in values_variable_2]
# plot_results(values_variable_2, results_variable_2, 'variable_2', 'Result')

# And so on...