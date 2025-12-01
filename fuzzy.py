import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Universos (faixas numéricas)
temperatura = ctrl.Antecedent(np.arange(15, 36, 1), "temperatura")
umidade = ctrl.Antecedent(np.arange(20, 91, 1), "umidade")
velocidade = ctrl.Consequent(np.arange(0, 101, 1), "velocidade")

# Funções de pertinência
temperatura["fria"]  = fuzz.trimf(temperatura.universe, [15, 18, 23])
temperatura["amena"] = fuzz.trimf(temperatura.universe, [20, 24, 28])
temperatura["quente"]= fuzz.trimf(temperatura.universe, [25, 30, 35])

umidade["seca"]   = fuzz.trimf(umidade.universe, [20, 30, 45])
umidade["normal"] = fuzz.trimf(umidade.universe, [40, 55, 70])
umidade["umida"]  = fuzz.trimf(umidade.universe, [65, 80, 90])

velocidade["baixa"] = fuzz.trimf(velocidade.universe, [0, 20, 45])
velocidade["media"] = fuzz.trimf(velocidade.universe, [30, 50, 70])
velocidade["alta"]  = fuzz.trimf(velocidade.universe, [60, 85, 100])

# Regras fuzzy
regras = [
    ctrl.Rule(temperatura["fria"]  & umidade["seca"],   velocidade["baixa"]),
    ctrl.Rule(temperatura["fria"]  & umidade["normal"], velocidade["baixa"]),
    ctrl.Rule(temperatura["fria"]  & umidade["umida"],  velocidade["media"]),

    ctrl.Rule(temperatura["amena"] & umidade["seca"],   velocidade["baixa"]),
    ctrl.Rule(temperatura["amena"] & umidade["normal"], velocidade["media"]),
    ctrl.Rule(temperatura["amena"] & umidade["umida"],  velocidade["alta"]),

    ctrl.Rule(temperatura["quente"] & umidade["seca"],   velocidade["media"]),
    ctrl.Rule(temperatura["quente"] & umidade["normal"], velocidade["alta"]),
    ctrl.Rule(temperatura["quente"] & umidade["umida"],  velocidade["alta"]),
]

sistema = ctrl.ControlSystem(regras)
simulador = ctrl.ControlSystemSimulation(sistema)


simulador.input["temperatura"] = 29
simulador.input["umidade"] = 78       

simulador.compute()

print("Velocidade recomendada do ventilador (%):", round(simulador.output["velocidade"], 2))

# (Opcional) visualizar a saída
velocidade.view(sim=simulador)
plt.ylabel("umidade")
plt.xlabel("temperatura")
plt.show()
