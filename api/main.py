from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

# Modelo de entrada para o Simplex
class SimplexInput(BaseModel):
    c: List[float]
    A: List[List[float]]
    b: List[float]
    tipo: str = 'max'  # Default 'max' para maximização

# Função Simplex
def simplex(c, A, b, tipo='max'):

    print('################################')
    print('  1° Etapa: Contrucao da Tabela')
    print('################################ \n')

    # Numero de restrições (linhas) e numero de variaveis (colunas)
    num_restricao, num_variaveis = A.shape

    # Criar a tabela inicial com as variaveis de folga
    tabela = np.zeros((num_restricao + 1, num_variaveis + num_restricao + 1))

    # Preencher a tabela com as restricoes
    tabela[:num_restricao, :num_variaveis] = A
    tabela[:num_restricao, num_variaveis:num_variaveis + num_restricao] = np.eye(num_restricao)
    tabela[:num_restricao, -1] = b

    # Preencher a linha da função objetivo
    if tipo == 'min':
        tabela[-1, :num_variaveis] = -c
    else:
        tabela[-1, :num_variaveis] = c

    # Função para verificar se o critério de otimalidade foi atingido
    def is_optimal():
        if tipo == 'min':
            return np.all(tabela[-1, :-1] >= 0)
        else:
            return np.all(tabela[-1, :-1] <= 0)

    # Funcao para escolher a variavel de entrada
    def get_pivot_column():
        if tipo == 'min':
            return np.argmin(tabela[-1, :-1])
        else:
            return np.argmax(tabela[-1, :-1])

    # Funcao para escolher a variavel de saida
    def get_pivot_row(pivot_col):
        ratios = tabela[:-1, -1] / tabela[:-1, pivot_col]
        valid_ratios = np.where(ratios > 0, ratios, np.inf)
        return np.argmin(valid_ratios)

    # Funcao de pivotagem
    def pivot(pivot_row, pivot_col):
        tabela[pivot_row, :] /= tabela[pivot_row, pivot_col]
        for i in range(len(tabela)):
            if i != pivot_row:
                tabela[i, :] -= tabela[i, pivot_col] * tabela[pivot_row, :]

    # Loop ate encontrar a solução otima
    interacao = 1
    while not is_optimal():
        print(f'Interação {interacao}:\n')
        pivot_col = get_pivot_column()
        pivot_row = get_pivot_row(pivot_col)
        pivot(pivot_row, pivot_col)
        interacao += 1
    print('Sim, terminou!!')

    # A solucao otima sera encontrada nas colunas das variaveis de folga
    solution = np.zeros(num_variaveis)
    for i in range(num_restricao):
        basic_var = np.where(np.abs(tabela[i, :num_variaveis]) == 1)[0]
        if len(basic_var) == 1:
            solution[basic_var[0]] = tabela[i, -1]

    return solution, tabela[-1, -1]

# Rota para resolver o problema com Simplex
@app.post("/solve_simplex/")
def solve_simplex(input_data: SimplexInput):
    try:
        c = np.array(input_data.c)
        A = np.array(input_data.A)
        b = np.array(input_data.b)
        solution, z_opt = simplex(c, A, b, tipo=input_data.tipo)
        return {"solution": solution.tolist(), "z_opt": z_opt}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Erro ao resolver o problema: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao resolver o problema.")
