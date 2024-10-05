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

# Função Simplex
def simplex(c, A, b):
    num_variables = len(c)
    num_constraints = len(b)

    # Verificar se o número de linhas de A é igual ao número de elementos em b
    assert len(A) == num_constraints, "A matriz A deve ter o mesmo número de linhas que o vetor b"

    # Criar a tabela do simplex
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    tableau[:num_constraints, :num_variables] = A
    tableau[:num_constraints, num_variables:num_variables + num_constraints] = np.eye(num_constraints)
    tableau[:num_constraints, -1] = b
    tableau[-1, :num_variables] = -c

    # Método Simplex
    while True:
        if np.all(tableau[-1, :-1] >= 0):
            break
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        basic_var = np.where(tableau[i, :num_variables] == 1)[0]
        if len(basic_var) == 1:
            solution[basic_var[0]] = tableau[i, -1]

    z_opt = tableau[-1, -1]
    return solution, z_opt

# Rota para resolver o problema com Simplex
@app.post("/solve_simplex/")
def solve_simplex(input_data: SimplexInput):
    try:
        # Printar os dados recebidos para depuração
        print(f"Recebido c: {input_data.c}")
        print(f"Recebido A: {input_data.A}")
        print(f"Recebido b: {input_data.b}")

        solution, z_opt = simplex(input_data.c, input_data.A, input_data.b)
        return {"solution": solution.tolist(), "z_opt": z_opt}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Printar erro para depuração
        print(f"Erro ao resolver o problema: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao resolver o problema.")
