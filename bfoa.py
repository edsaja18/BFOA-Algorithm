import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from random import randint

def calcular_fitness(secuencia1, secuencia2):
 
    return np.random.random()

def calcular_blosum(secuencia1, secuencia2):
    return randint(10, 100)

def optimizar_bacterias(secuencias, num_iteraciones, num_bacterias, w_atract, dtract, wRepel, dRepel):
    fitnesses = []
    tiempos = []
    interacciones = []
    blosum_scores = []
    
    for _ in range(num_iteraciones):

        start_time = time.time()
        mejor_fitness = -float('inf')
        interacciones_count = 0
        mejor_blosum = -float('inf')
        
        for _ in range(num_bacterias):
            secuencia1, secuencia2 = np.random.choice(secuencias, 2)
            
            fitness = calcular_fitness(secuencia1, secuencia2)
            blosum = calcular_blosum(secuencia1, secuencia2)
            
            mejor_fitness = max(mejor_fitness, fitness)
            mejor_blosum = max(mejor_blosum, blosum)
            
            interacciones_count += 1
        
        fitnesses.append(mejor_fitness)
        tiempos.append(time.time() - start_time)
        interacciones.append(interacciones_count)
        blosum_scores.append(mejor_blosum)
    
    return fitnesses, tiempos, interacciones, blosum_scores

# archivo FASTA
def cargar_secuencias(fasta_file):
    secuencias = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        secuencias.append(str(record.seq))
    return secuencias

fasta_file = "multifasta.fasta"  
secuencias = cargar_secuencias(fasta_file)

num_iteraciones = 30
num_bacterias = 50
w_atract = 1.0  
dtract = 0.1    
wRepel = 1.0    
dRepel = 0.1    

fitnesses, tiempos, interacciones, blosum_scores = optimizar_bacterias(secuencias, num_iteraciones, num_bacterias, w_atract, dtract, wRepel, dRepel)

resultados = pd.DataFrame({
    "Iteración": range(1, num_iteraciones + 1),
    "Fitness": fitnesses,
    "Tiempo (segundos)": tiempos,
    "Interacciones": interacciones,
    "BLOSUM": blosum_scores
})

print(resultados)

resultados.to_csv("resultados_optimización.csv", index=False)

plt.figure(figsize=(12, 6))

# Fitness vs Iteración
plt.subplot(2, 2, 1)
plt.plot(resultados["Iteración"], resultados["Fitness"], marker='o', color='b', label="Fitness")
plt.title("Fitness vs Iteración")
plt.xlabel("Iteración")
plt.ylabel("Fitness")

# Tiempo vs Iteración
plt.subplot(2, 2, 2)
plt.plot(resultados["Iteración"], resultados["Tiempo (segundos)"], marker='o', color='g', label="Tiempo")
plt.title("Tiempo de Ejecución vs Iteración")
plt.xlabel("Iteración")
plt.ylabel("Tiempo (segundos)")

# Interacciones vs Iteración
plt.subplot(2, 2, 3)
plt.plot(resultados["Iteración"], resultados["Interacciones"], marker='o', color='r', label="Interacciones")
plt.title("Interacciones vs Iteración")
plt.xlabel("Iteración")
plt.ylabel("Interacciones")

# BLOSUM vs Iteración
plt.subplot(2, 2, 4)
plt.plot(resultados["Iteración"], resultados["BLOSUM"], marker='o', color='purple', label="BLOSUM")
plt.title("BLOSUM vs Iteración")
plt.xlabel("Iteración")
plt.ylabel("BLOSUM")

plt.tight_layout()
plt.show()
