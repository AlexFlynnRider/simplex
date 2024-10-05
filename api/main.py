from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np

app = FastAPI()

# Modelo de entrada para o Simplex
class SimplexInput(BaseModel):
    c: List[float]
    A: List[List[float]]
    b: List[float]
    tipo: str = 'max'  # Default 'max' para maximização

# Função Simplex
def simplex(c, A, b, tipo='max') -> Tuple[np.ndarray, float, List[str]]:
    logs = []  # Lista para armazenar as mensagens de log

    logs.append('################################')
    logs.append('  1° Etapa: Contrucao da Tabela')
    logs.append('################################ \n')

    # Numero de restrições (linhas) e numero de variaveis (colunas)
    num_restricao, num_variaveis = A.shape

    # Criar a tabela inicial com as variaveis de folga
    tabela = np.zeros((num_restricao + 1, num_variaveis + num_restricao + 1))

    # Preencher a tabela com as restricoes
    tabela[:num_restricao, :num_variaveis] = A
    tabela[:num_restricao, num_variaveis:num_variaveis + num_restricao] = np.eye(num_restricao)
    tabela[:num_restricao, -1] = b

    logs.append(f"Passo 1: Tabela inicial criada:\n{tabela}\n")

    # Preencher a linha da função objetivo
    if tipo == 'min':
        tabela[-1, :num_variaveis] = -c
    else:
        tabela[-1, :num_variaveis] = c

    logs.append(f"Passo 2: Função objetivo adicionada:\n{tabela}\n")

    # Função para verificar se o critério de otimalidade foi atingido
    def is_optimal():
        if tipo == 'min':
            logs.append(f"Critério de Otimalidade (Minimização): Verificando se todos os coeficientes são >= 0")
            return np.all(tabela[-1, :-1] >= 0)
        else:
            logs.append(f"Critério de Otimalidade (Maximização): Verificando se todos os coeficientes são <= 0")
            return np.all(tabela[-1, :-1] <= 0)

    # Funcao para escolher a variavel de entrada
    def get_pivot_column():
        if tipo == 'min':
            coluna = np.argmin(tabela[-1, :-1])
            logs.append(f"Coluna Pivô (Minimização): Coluna {coluna + 1} com menor valor na função objetivo")
            return coluna
        else:
            coluna = np.argmax(tabela[-1, :-1])
            logs.append(f"Coluna Pivô (Maximização): Coluna {coluna + 1} com maior valor na função objetivo")
            return coluna

    # Funcao para escolher a variavel de saida
    def get_pivot_row(pivot_col):
        ratios = tabela[:-1, -1] / tabela[:-1, pivot_col]
        valid_ratios = np.where(ratios > 0, ratios, np.inf)
        pivot_row = np.argmin(valid_ratios)
        logs.append(f"Razão mínima calculada. Linha pivô: {pivot_row + 1}")
        return pivot_row

    # Funcao de pivotagem
    def pivot(pivot_row, pivot_col):
        logs.append(f"Pivotando na linha {pivot_row + 1}, coluna {pivot_col + 1}")
        tabela[pivot_row, :] /= tabela[pivot_row, pivot_col]
        for i in range(len(tabela)):
            if i != pivot_row:
                tabela[i, :] -= tabela[i, pivot_col] * tabela[pivot_row, :]
        logs.append(f"Tabela atualizada após pivotagem:\n{tabela}\n")

    # Loop ate encontrar a solução otima
    interacao = 1
    while not is_optimal():
        logs.append(f'Interação {interacao}:\n')
        pivot_col = get_pivot_column()
        pivot_row = get_pivot_row(pivot_col)
        pivot(pivot_row, pivot_col)
        interacao += 1

    logs.append('Solução ótima encontrada!\n')

    # A solucao otima sera encontrada nas colunas das variaveis de folga
    solution = np.zeros(num_variaveis)
    for i in range(num_restricao):
        basic_var = np.where(np.abs(tabela[i, :num_variaveis]) == 1)[0]
        if len(basic_var) == 1:
            solution[basic_var[0]] = tabela[i, -1]

    logs.append(f"Solução ótima: {solution}")
    logs.append(f"Valor ótimo: {tabela[-1, -1]}")

    return solution, tabela[-1, -1], logs

# Rota para resolver o problema com Simplex
@app.post("/solve_simplex/")
def solve_simplex(input_data: SimplexInput):
    try:
        c = np.array(input_data.c)
        A = np.array(input_data.A)
        b = np.array(input_data.b)
        solution, z_opt, logs = simplex(c, A, b, tipo=input_data.tipo)
        return {"solution": solution.tolist(), "z_opt": z_opt, "logs": logs}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Erro ao resolver o problema: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao resolver o problema.")
