# Guia Completo: Do Zero à Inteligência Artificial
## Um Caminho Prático para Programadores em Evolução

---

## PARTE 1: FUNDAMENTOS ESSENCIAIS

### Capítulo 1: Conceitos Básicos que Você Precisa Entender

#### 1.1 O que é Inteligência Artificial?

Inteligência Artificial (IA) é a capacidade de um programa tomar decisões ou fazer previsões com base em padrões encontrados em dados. Diferente de um programa tradicional onde você escreve todas as regras, em IA o próprio programa aprende as regras a partir de exemplos.

**Exemplo prático:** Um programa tradicional para classificar e-mails como spam precisaria de milhares de regras escritas por humanos. Um programa de IA recebe exemplos de e-mails spam e aprende sozinho quais características indicam spam.

#### 1.2 Os Três Pilares da IA Moderna

**Machine Learning (ML):** O programa aprende através de exemplos. Você fornece dados e o algoritmo descobre padrões.

**Deep Learning (DL):** Um tipo especial de Machine Learning que usa estruturas chamadas Redes Neurais, inspiradas no cérebro humano.

**Processamento de Linguagem Natural (NLP):** Especialização que ensina ao computador a entender e processar linguagem humana.

#### 1.3 Matemática Essencial (Não Se Assuste!)

Você só precisa entender três conceitos:

**Álgebra Linear:** Matrizes e vetores. Pense em matrizes como tabelas de números que armazenam seus dados.

```
Matriz exemplo:
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
```

**Cálculo:** Como uma quantidade muda em relação a outra. Essencial para "treinar" modelos.

**Probabilidade:** A chance de algo acontecer. Usado para fazer previsões.

Não precisa dominar tudo agora. Aprenderemos conforme precisarmos na prática.

#### 1.4 Preparando Seu Ambiente

Você precisará de:

**Python:** Linguagem padrão para IA. Fácil de aprender e poderosa.

**Bibliotecas essenciais:**
- NumPy: Para trabalhar com matrizes e operações matemáticas
- Pandas: Para manipular dados
- Matplotlib: Para visualizar dados
- Scikit-learn: Para algoritmos de Machine Learning

Instalação (após ter Python):
```
pip install numpy pandas matplotlib scikit-learn
```

---

### Capítulo 2: Preparando Dados para IA

#### 2.1 Por Que Dados São Tudo

Um modelo de IA é tão bom quanto os dados que o treinam. Dados ruins = modelo ruim.

#### 2.2 Entendendo Dados

Cada exemplo de dado possui características (features) e um resultado esperado (label).

**Exemplo - Previsão de Preço de Imóvel:**

| Quartos | Área (m²) | Idade | Preço (R$) |
|---------|-----------|-------|-----------|
| 3 | 100 | 10 | 350.000 |
| 4 | 150 | 5 | 500.000 |
| 2 | 80 | 20 | 250.000 |

Características: Quartos, Área, Idade
Resultado esperado (label): Preço

#### 2.3 Processando Dados em Python

```python
import pandas as pd
import numpy as np

# Carregar dados (simulando)
dados = {
    'quartos': [3, 4, 2, 3, 5],
    'area': [100, 150, 80, 110, 200],
    'idade': [10, 5, 20, 8, 2],
    'preco': [350000, 500000, 250000, 380000, 600000]
}

df = pd.DataFrame(dados)

# Verificar dados
print(df.head())  # Primeiras linhas
print(df.describe())  # Estatísticas

# Dividir em características (X) e resultados (y)
X = df[['quartos', 'area', 'idade']]
y = df['preco']
```

#### 2.4 Limpeza e Normalização

**Limpeza:** Remover dados inválidos ou faltantes

```python
# Remover linhas com valores faltantes
df = df.dropna()

# Remover duplicatas
df = df.drop_duplicates()
```

**Normalização:** Colocar todos os valores em uma escala similar (0 a 1)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)
```

#### 2.5 Dividindo Dados para Treinamento

Nunca treine e teste com os mesmos dados! É como estudar com o gabarito aberto.

```python
from sklearn.model_selection import train_test_split

# 80% treino, 20% teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### 2.6 Caso de Uso: Dataset Iris

Você também pode usar datasets prontos para praticar:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Características
y = iris.target  # Classificação (tipo de flor)

print(f"Amostras: {X.shape[0]}")
print(f"Características por amostra: {X.shape[1]}")
```

---

## PARTE 2: REDES NEURAIS

### Capítulo 3: Entendendo Redes Neurais

#### 3.1 Como Funciona o Cérebro Biológico

O cérebro tem neurônios. Cada neurônio recebe sinais de outros neurônios, processa e envia para frente. Redes Neurais Artificiais imitam esse processo.

#### 3.2 O Neurônio Artificial

Um neurônio artificial recebe múltiplas entradas, multiplica cada uma por um peso, soma tudo, passa por uma função de ativação e gera uma saída.

**Fórmula:**
```
saida = funcao_ativacao(w1*x1 + w2*x2 + w3*x3 + bias)
```

Onde:
- x1, x2, x3: entradas
- w1, w2, w3: pesos (ajustados durante treinamento)
- bias: valor de ajuste
- funcao_ativacao: função que adiciona não-linearidade

#### 3.3 Estrutura de uma Rede Neural

Uma Rede Neural possui camadas:

**Camada de Entrada:** Recebe os dados brutos

**Camadas Ocultas:** Processam informações, encontram padrões

**Camada de Saída:** Fornece a previsão final

```
Entrada → Camada Oculta 1 → Camada Oculta 2 → Saída
  (4)         (8 neurônios)    (4 neurônios)    (1)
```

#### 3.4 Funções de Ativação

As funções de ativação adicionam não-linearidade, permitindo que a rede aprenda padrões complexos.

**ReLU (Rectified Linear Unit):** A mais comum
```python
def relu(x):
    return max(0, x)
```

**Sigmoid:** Para classificações
```python
def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

#### 3.5 Treinamento: Como a Rede Aprende

O treinamento funciona assim:

1. **Forward Pass:** Os dados passam pela rede, gerando uma previsão
2. **Calcular Erro:** Comparar previsão com resultado esperado
3. **Backward Pass:** Calcular como cada peso contribuiu para o erro
4. **Atualizar Pesos:** Ajustar os pesos para reduzir o erro
5. **Repetir:** Fazer isso muitas vezes até os pesos ficarem bons

Esse processo é chamado **Propagação para Trás (Backpropagation)**.

### Capítulo 4: Implementando Sua Primeira Rede Neural do Zero

#### 4.1 Construindo Neurônio Simples

```python
import numpy as np

class Neuronio:
    def __init__(self, num_entradas):
        # Inicializar pesos aleatoriamente
        self.pesos = np.random.randn(num_entradas) * 0.01
        self.bias = 0
    
    def ativar(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def processar(self, entradas):
        """Forward pass"""
        # Multiplicar entradas por pesos e somar bias
        z = np.dot(entradas, self.pesos) + self.bias
        # Aplicar função de ativação
        saida = self.ativar(z)
        return saida
```

#### 4.2 Construindo Camada Neural

```python
class Camada:
    def __init__(self, num_entradas, num_neuronios):
        self.neuronios = [Neuronio(num_entradas) for _ in range(num_neuronios)]
    
    def processar(self, entradas):
        """Processar dados através de todos os neurônios"""
        saidas = []
        for neuronio in self.neuronios:
            saidas.append(neuronio.processar(entradas))
        return np.array(saidas)
```

#### 4.3 Construindo Rede Neural Completa

```python
class RedeNeural:
    def __init__(self, arquitetura):
        """
        arquitetura: lista com quantidade de neurônios por camada
        Ex: [4, 8, 4, 1] = entrada com 4 features, 
                          camada oculta 1 com 8, 
                          camada oculta 2 com 4,
                          saída com 1
        """
        self.camadas = []
        
        for i in range(len(arquitetura) - 1):
            camada = Camada(arquitetura[i], arquitetura[i + 1])
            self.camadas.append(camada)
    
    def prever(self, entradas):
        """Fazer previsão para novos dados"""
        dados_atuais = entradas
        
        for camada in self.camadas:
            dados_atuais = camada.processar(dados_atuais)
        
        return dados_atuais
```

#### 4.4 Testando Sua Rede

```python
# Criar rede: 4 entradas -> 8 ocultos -> 4 ocultos -> 1 saída
rede = RedeNeural([4, 8, 4, 1])

# Dados de teste
entrada = np.array([0.5, 0.2, 0.8, 0.3])

# Fazer previsão
saida = rede.prever(entrada)
print(f"Previsão: {saida}")
```

#### 4.5 Usando TensorFlow/Keras (Simples)

Para aplicações reais, use bibliotecas prontas:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Criar modelo
modelo = keras.Sequential([
    layers.Input(shape=(4,)),  # 4 features de entrada
    layers.Dense(8, activation='relu'),  # Camada oculta 1
    layers.Dense(4, activation='relu'),  # Camada oculta 2
    layers.Dense(1, activation='sigmoid')  # Saída
])

# Compilar
modelo.compile(optimizer='adam', loss='binary_crossentropy')

# Treinar
modelo.fit(X_treino, y_treino, epochs=10, batch_size=32)

# Prever
previsoes = modelo.predict(X_teste)
```

#### 4.6 Caso de Uso: Classificar Iris

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Carregar dados
iris = load_iris()
X = iris.data
y = iris.target

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)

# Criar modelo
modelo = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
               metrics=['accuracy'])

# Treinar
modelo.fit(X_treino, y_treino, epochs=50, verbose=0)

# Avaliar
loss, acuracia = modelo.evaluate(X_teste, y_teste)
print(f"Acurácia: {acuracia:.2%}")
```

---

## PARTE 3: ALGORITMOS GENÉTICOS

### Capítulo 5: Entendendo Algoritmos Genéticos

#### 5.1 A Inspiração: Evolução Natural

Algoritmos Genéticos simulam evolução biológica. Assim como na natureza, os "indivíduos" mais adaptados sobrevivem e se reproduzem, gerando descendentes com características melhoradas.

#### 5.2 Conceitos Fundamentais

**Gene:** Um valor que representa uma característica

**Cromossomo:** Uma sequência de genes (a solução completa)

**População:** Conjunto de cromossomos

**Fitness:** Quanto um cromossomo é bom (quão perto ele está da solução)

**Mutação:** Mudança aleatória em um gene

**Crossover:** Combinar dois cromossomos para criar filhos

#### 5.3 O Algoritmo em Passos

1. Criar população aleatória
2. Calcular fitness de cada indivíduo
3. Selecionar os melhores
4. Aplicar crossover nos melhores
5. Aplicar mutação
6. Repetir até convergência

### Capítulo 6: Implementando Algoritmo Genético

#### 6.1 Versão Simples: Encontrar um Número

```python
import random
import numpy as np

class AlgoritmoGenetico:
    def __init__(self, alvo, tamanho_populacao=100, taxa_mutacao=0.1):
        self.alvo = alvo
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.populacao = [random.random() * 100 for _ in range(tamanho_populacao)]
    
    def calcular_fitness(self, individuo):
        """Quanto mais perto do alvo, melhor o fitness"""
        distancia = abs(individuo - self.alvo)
        fitness = 1 / (1 + distancia)  # Normalizar entre 0 e 1
        return fitness
    
    def selecionar_pais(self):
        """Selecionar os melhores indivíduos"""
        fitness_scores = [self.calcular_fitness(ind) for ind in self.populacao]
        
        # Selecionar os 50% melhores
        indices = np.argsort(fitness_scores)[-self.tamanho_populacao // 2:]
        return [self.populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        """Combinar dois pais"""
        # Média simples
        filho = (pai1 + pai2) / 2
        return filho
    
    def mutar(self, individuo):
        """Aplicar mutação aleatória"""
        if random.random() < self.taxa_mutacao:
            individuo += random.uniform(-5, 5)
        return individuo
    
    def evolucionar(self, gerações=100):
        """Executar o algoritmo genético"""
        for geracao in range(gerações):
            pais = self.selecionar_pais()
            nova_populacao = []
            
            while len(nova_populacao) < self.tamanho_populacao:
                pai1, pai2 = random.sample(pais, 2)
                filho = self.crossover(pai1, pai2)
                filho = self.mutar(filho)
                nova_populacao.append(filho)
            
            self.populacao = nova_populacao
            
            melhor = max(self.populacao, key=self.calcular_fitness)
            if geracao % 10 == 0:
                print(f"Geração {geracao}: Melhor = {melhor:.4f}, Alvo = {self.alvo}")
        
        return max(self.populacao, key=self.calcular_fitness)

# Usar
ag = AlgoritmoGenetico(alvo=42.5)
resultado = ag.evolucionar(gerações=100)
print(f"Resultado final: {resultado:.4f}")
```

#### 6.2 Otimizando Funções Complexas

```python
class AGFuncao:
    def __init__(self, funcao, limites, tamanho_populacao=50):
        """
        funcao: função a otimizar
        limites: lista de tuplas [(min1, max1), (min2, max2), ...]
        """
        self.funcao = funcao
        self.limites = limites
        self.tamanho_populacao = tamanho_populacao
        self.populacao = self.criar_populacao_inicial()
    
    def criar_populacao_inicial(self):
        populacao = []
        for _ in range(self.tamanho_populacao):
            individuo = [random.uniform(lim[0], lim[1]) for lim in self.limites]
            populacao.append(individuo)
        return populacao
    
    def calcular_fitness(self, individuo):
        """Quanto maior, melhor"""
        return self.funcao(individuo)
    
    def selecionar(self):
        fitness_scores = [self.calcular_fitness(ind) for ind in self.populacao]
        indices = np.argsort(fitness_scores)[-len(self.populacao)//2:]
        return [self.populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        filho = [(pai1[i] + pai2[i]) / 2 for i in range(len(pai1))]
        return filho
    
    def mutar(self, individuo, taxa=0.1):
        for i in range(len(individuo)):
            if random.random() < taxa:
                individuo[i] += random.uniform(-0.5, 0.5)
                individuo[i] = np.clip(individuo[i], self.limites[i][0], self.limites[i][1])
        return individuo
    
    def evoluir(self, gerações=100):
        for gen in range(gerações):
            pais = self.selecionar()
            nova_pop = []
            
            while len(nova_pop) < self.tamanho_populacao:
                pai1, pai2 = random.sample(pais, 2)
                filho = self.crossover(pai1, pai2)
                filho = self.mutar(filho)
                nova_pop.append(filho)
            
            self.populacao = nova_pop
        
        melhor = max(self.populacao, key=self.calcular_fitness)
        return melhor, self.calcular_fitness(melhor)

# Exemplo: Encontrar máximo de função
def funcao_teste(x):
    """Encontrar máximo: -x² é máximo em x=0"""
    return -(x[0]**2 + x[1]**2)

ag = AGFuncao(funcao_teste, limites=[(-10, 10), (-10, 10)])
melhor, fitness = ag.evoluir(gerações=50)
print(f"Melhor solução: {melhor}, Fitness: {fitness}")
```

#### 6.3 Caso de Uso: Problema do Caixeiro Viajante (TSP)

```python
import math

class TSPGenetico:
    def __init__(self, cidades, tamanho_pop=50):
        """
        cidades: lista de tuplas (x, y)
        """
        self.cidades = cidades
        self.tamanho_pop = tamanho_pop
        self.num_cidades = len(cidades)
        self.populacao = self.criar_populacao()
    
    def distancia(self, cidade1, cidade2):
        return math.sqrt((cidade1[0]-cidade2[0])**2 + (cidade1[1]-cidade2[1])**2)
    
    def calcular_distancia_rota(self, rota):
        dist_total = 0
        for i in range(len(rota)):
            cidade_atual = rota[i]
            cidade_proxima = rota[(i + 1) % len(rota)]
            dist_total += self.distancia(self.cidades[cidade_atual], 
                                        self.cidades[cidade_proxima])
        return dist_total
    
    def criar_populacao(self):
        populacao = []
        for _ in range(self.tamanho_pop):
            rota = list(range(self.num_cidades))
            random.shuffle(rota)
            populacao.append(rota)
        return populacao
    
    def selecionar(self):
        distancias = [self.calcular_distancia_rota(rota) for rota in self.populacao]
        indices = np.argsort(distancias)[:self.tamanho_pop//2]
        return [self.populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        tamanho = len(pai1)
        ponto1, ponto2 = sorted(random.sample(range(tamanho), 2))
        
        filho = [-1] * tamanho
        filho[ponto1:ponto2] = pai1[ponto1:ponto2]
        
        idx = ponto2
        for gene in pai2:
            if gene not in filho:
                filho[idx % tamanho] = gene
                idx += 1
        
        return filho
    
    def mutar(self, rota, taxa=0.1):
        if random.random() < taxa:
            i, j = random.sample(range(len(rota)), 2)
            rota[i], rota[j] = rota[j], rota[i]
        return rota
    
    def evoluir(self, gerações=100):
        for gen in range(gerações):
            pais = self.selecionar()
            nova_pop = []
            
            while len(nova_pop) < self.tamanho_pop:
                pai1, pai2 = random.sample(pais, 2)
                filho = self.crossover(pai1, pai2)
                filho = self.mutar(filho)
                nova_pop.append(filho)
            
            self.populacao = nova_pop
            
            if gen % 20 == 0:
                melhor = min(self.populacao, 
                           key=self.calcular_distancia_rota)
                dist = self.calcular_distancia_rota(melhor)
                print(f"Geração {gen}: Melhor distância = {dist:.2f}")
        
        melhor_rota = min(self.populacao, key=self.calcular_distancia_rota)
        return melhor_rota, self.calcular_distancia_rota(melhor_rota)

# Usar
cidades = [(0, 0), (10, 5), (15, 10), (5, 15), (2, 8)]
tsp = TSPGenetico(cidades)
rota, distancia = tsp.evoluir(gerações=100)
print(f"Melhor rota: {rota}, Distância: {distancia:.2f}")
```

---

## PARTE 4: PROCESSAMENTO DE LINGUAGEM NATURAL (NLP)

### Capítulo 7: Fundamentos de NLP

#### 7.1 O Desafio de Processar Linguagem

Computadores trabalham com números, não com palavras. NLP converte linguagem humana em números que máquinas conseguem processar.

#### 7.2 Conceitos Básicos

**Tokenização:** Dividir texto em palavras (tokens)

```
"Olá mundo" → ["Olá", "mundo"]
```

**Limpeza:** Remover pontuação, converter para minúscula, etc.

```
"Olá, Mundo!" → ["olá", "mundo"]
```

**Stemming/Lemmatização:** Reduzir palavras à raiz

```
"correndo", "corre", "corrida" → "corr" (stem) ou "correr" (lemma)
```

**Bag of Words:** Contar frequência de cada palavra

```
"gato gato cão" → {'gato': 2, 'cão': 1}
```

### Capítulo 8: Implementando NLP do Zero

#### 8.1 Processador de Texto Básico

```python
import re
from collections import Counter

class ProcessadorTexto:
    def __init__(self):
        self.palavras_parada = {
            'o', 'a', 'de', 'para', 'com', 'é', 'e', 'que', 'um', 'uma'
        }
    
    def limpar_texto(self, texto):
        """Limpeza básica"""
        # Converter para minúscula
        texto = texto.lower()
        # Remover pontuação
        texto = re.sub(r'[^\w\s]', '', texto)
        # Remover números
        texto = re.sub(r'\d+', '', texto)
        return texto
    
    def tokenizar(self, texto):
        """Dividir em palavras"""
        texto_limpo = self.limpar_texto(texto)
        return texto_limpo.split()
    
    def remover_parada(self, tokens):
        """Remover palavras comuns sem significado"""
        return [t for t in tokens if t not in self.palavras_parada]
    
    def processar(self, texto):
        """Pipeline completo"""
        tokens = self.tokenizar(texto)
        tokens = self.remover_parada(tokens)
        return tokens

# Usar
processador = ProcessadorTexto()
texto = "Olá mundo! Este é um exemplo de processamento de linguagem natural."
tokens = processador.processar(texto)
print(tokens)
```

#### 8.2 Análise de Sentimentos

```python
class AnalisadorSentimento:
    def __init__(self):
        self.positivas = {'bom', 'ótimo', 'excelente', 'amor', 'feliz', 'adorei'}
        self.negativas = {'ruim', 'horrível', 'péssimo', 'ódio', 'triste', 'odiei'}
        self.processador = ProcessadorTexto()
    
    def analisar(self, texto):
        """Retorna -1 (negativo), 0 (neutro), 1 (positivo)"""
        tokens = self.processador.processar(texto)
        
        positivas = sum(1 for t in tokens if t in self.positivas)
        negativas = sum(1 for t in tokens if t in self.negativas)
        
        if positivas > negativas:
            return 1, "Positivo"
        elif negativas > positivas:
            return -1, "Negativo"
        else:
            return 0, "Neutro"

# Usar
analisador = AnalisadorSentimento()
resultado, label = analisador.analisar("Adorei o produto, é excelente!")
print(f"Sentimento: {label}")
```

#### 8.3 Bag of Words e TF-IDF

```python
class VectorizadorTexto:
    def __init__(self):
        self.processador = ProcessadorTexto()
        self.vocabulario = {}
    
    def construir_vocabulario(self, documentos):
        """Criar vocabulário único"""
        palavra_id = 0
        for doc in documentos:
            tokens = self.processador.processar(doc)
            for token in tokens:
                if token not in self.vocabulario:
                    self.vocabulario[token] = palavra_id
                    palavra_id += 1
    
    def bag_of_words(self, texto):
        """Converter texto em vetor de frequências"""
        tokens = self.processador.processar(texto)
        vetor = [0] * len(self.vocabulario)
        
        for token in tokens:
            if token in self.vocabulario:
                idx = self.vocabulario[token]
                vetor[idx] += 1
        
        return vetor

# Usar
vectorizador = VectorizadorTexto()
documentos = [
    "gato gato cão",
    "passarinho voa alto",
    "gato e cão brigam"
]

vectorizador.construir_vocabulario(documentos)
vetor = vectorizador.bag_of_words("gato cão")
print(f"Vocabulário: {vectorizador.vocabulario}")
print(f"Vetor: {vetor}")
```

#### 8.4 Classificador de Texto com IA

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class ClassificadorTexto:
    def __init__(self):
        self.vectorizador = TfidfVectorizer()
        self.modelo = MultinomialNB()
    
    def treinar(self, textos, labels):
        """
        textos: lista de strings
        labels: lista de categorias
        """
        X = self.vectorizador.fit_transform(textos)
        self.modelo.fit(X, labels)
    
    def classificar(self, texto):
        """Classificar novo texto"""
        X = self.vectorizador.transform([texto])
        return self.modelo.predict(X)[0]

# Usar
classificador = ClassificadorTexto()

textos_treino = [
    "gato é um animal fofo",
    "gato gosta de peixe",
    "cão é o melhor amigo",
    "cão gosta de brincar"
]

labels_treino = ["animal", "animal", "animal", "animal"]

classificador.treinar(textos_treino, labels_treino)
resultado = classificador.classificar("gato come peixe")
print(f"Classificação: {resultado}")
```

#### 8.5 Caso de Uso: Detector de Spam

```python
class DetectorSpam:
    def __init__(self):
        self.processador = ProcessadorTexto()
        self.palavras_spam = {
            'grátis', 'prêmio', 'clique', 'urgente', 'ganhe', 'oferta',
            'desconto', 'promoção', 'exclusivo', 'limitado'
        }
    
    def calcular_score_spam(self, texto):
        """Retorna score entre 0 e 1"""
        tokens = self.processador.processar(texto)
        
        if not tokens:
            return 0
        
        palavras_spam_encontradas = sum(1 for t in tokens if t in self.palavras_spam)
        score = palavras_spam_encontradas / len(tokens)
        
        return score
    
    def classificar(self, texto, limiar=0.3):
        """Classifica como spam ou não"""
        score = self.calcular_score_spam(texto)
        eh_spam = score >= limiar
        return eh_spam, score

# Usar
detector = DetectorSpam()

emails = [
    "Olá, tudo bem com você?",
    "GANHE PRÊMIO GRÁTIS! CLIQUE AGORA!",
    "Reunião amanhã às 10h",
    "OFERTA EXCLUSIVA! DESCONTO LIMITADO!"
]

for email in emails:
    eh_spam, score = detector.classificar(email)
    print(f"Email: {email[:30]}... | Spam: {eh_spam} (Score: {score:.2f})")
```

---

## PARTE 5: PROJETO FINAL INTEGRADO

### Capítulo 9: Projeto Completo - Sistema de Recomendação de Plantas

Vamos construir um sistema que:
1. **Usa uma Rede Neural** para prever se uma planta vai sobreviver
2. **Usa Algoritmo Genético** para otimizar condições ideais
3. **Usa NLP** para processar descrições de plantas
4. **Integra tudo** em um software funcional

#### 9.1 Estrutura do Projeto

```
sistema_plantas/
├── dados.py          # Gerencia dados
├── rede_neural.py    # Modelo de IA
├── genetico.py       # Otimizador
├── nlp.py            # Processamento de texto
└── main.py           # Orquestrador
```

#### 9.2 Módulo de Dados

```python
# dados.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GerenciadorDados:
    def __init__(self):
        self.dados = None
        self.scaler = StandardScaler()
    
    def carregar_dados_teste(self):
        """Criar dataset de exemplo de plantas"""
        dados = {
            'luz_horas': [8, 6, 10, 5, 9, 7, 11, 4],
            'agua_ml': [500, 300, 700, 200, 600, 400, 800, 150],
            'temperatura': [25, 20, 28, 18, 26, 22, 30, 16],
            'umidade': [60, 50, 70, 45, 65, 55, 75, 40],
            'sobrevive': [1, 1, 1, 0, 1, 1, 1, 0]
        }
        self.dados = pd.DataFrame(dados)
        return self.dados
    
    def preparar_dados(self):
        """Normalizar e dividir dados"""
        X = self.dados[['luz_horas', 'agua_ml', 'temperatura', 'umidade']]
        y = self.dados['sobrevive']
        
        X_normalizado = self.scaler.fit_transform(X)
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X_normalizado, y, test_size=0.2, random_state=42
        )
        
        return X_treino, X_teste, y_treino, y_teste
```

#### 9.3 Módulo de Rede Neural

```python
# rede_neural.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

class ModeloPlantas:
    def __init__(self):
        self.modelo = None
    
    def construir(self):
        """Construir arquitetura da rede"""
        self.modelo = keras.Sequential([
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def treinar(self, X_treino, y_treino, X_teste, y_teste):
        """Treinar o modelo"""
        self.modelo.fit(
            X_treino, y_treino,
            epochs=50,
            batch_size=2,
            validation_data=(X_teste, y_teste),
            verbose=0
        )
    
    def prever(self, condicoes):
        """Prever se planta sobrevive"""
        previsao = self.modelo.predict(np.array([condicoes]), verbose=0)
        return previsao[0][0]
    
    def avaliar(self, X_teste, y_teste):
        """Avaliar desempenho"""
        loss, accuracy = self.modelo.evaluate(X_teste, y_teste, verbose=0)
        return accuracy
```

#### 9.4 Módulo de Algoritmo Genético

```python
# genetico.py
import random
import numpy as np

class OtimizadorGenetico:
    def __init__(self, modelo_rede, limites):
        """
        modelo_rede: instância de ModeloPlantas
        limites: dict com ranges {'luz': (min, max), ...}
        """
        self.modelo = modelo_rede
        self.limites = limites
        self.populacao = self.criar_populacao(30)
    
    def criar_populacao(self, tamanho):
        """Criar condições aleatórias"""
        populacao = []
        for _ in range(tamanho):
            condicoes = {
                'luz': random.uniform(self.limites['luz'][0], self.limites['luz'][1]),
                'agua': random.uniform(self.limites['agua'][0], self.limites['agua'][1]),
                'temp': random.uniform(self.limites['temp'][0], self.limites['temp'][1]),
                'umidade': random.uniform(self.limites['umidade'][0], self.limites['umidade'][1])
            }
            populacao.append(condicoes)
        return populacao
    
    def calcular_fitness(self, condicoes):
        """Quão bom é este conjunto de condições"""
        vetor = np.array([condicoes['luz'], condicoes['agua'], 
                         condicoes['temp'], condicoes['umidade']])
        probabilidade_sobrevivencia = self.modelo.prever(vetor)
        return probabilidade_sobrevivencia
    
    def selecionar(self):
        """Selecionar melhores indivíduos"""
        fitness_scores = [self.calcular_fitness(c) for c in self.populacao]
        indices = np.argsort(fitness_scores)[-len(self.populacao)//2:]
        return [self.populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        """Combinar dois pais"""
        filho = {
            'luz': (pai1['luz'] + pai2['luz']) / 2,
            'agua': (pai1['agua'] + pai2['agua']) / 2,
            'temp': (pai1['temp'] + pai2['temp']) / 2,
            'umidade': (pai1['umidade'] + pai2['umidade']) / 2
        }
        return filho
    
    def mutar(self, condicoes, taxa=0.1):
        """Aplicar mutação"""
        if random.random() < taxa:
            chave = random.choice(list(condicoes.keys()))
            condicoes[chave] += random.uniform(-2, 2)
            condicoes[chave] = np.clip(
                condicoes[chave],
                self.limites[chave][0],
                self.limites[chave][1]
            )
        return condicoes
    
    def evoluir(self, gerações=20):
        """Executar otimização"""
        for gen in range(gerações):
            pais = self.selecionar()
            nova_populacao = []
            
            while len(nova_populacao) < len(self.populacao):
                pai1, pai2 = random.sample(pais, 2)
                filho = self.crossover(pai1, pai2)
                filho = self.mutar(filho)
                nova_populacao.append(filho)
            
            self.populacao = nova_populacao
        
        melhor = max(self.populacao, key=self.calcular_fitness)
        return melhor
```

#### 9.5 Módulo de NLP

```python
# nlp.py
import re
from collections import Counter

class ProcessadorDescricoes:
    def __init__(self):
        self.palavras_parada = {
            'o', 'a', 'de', 'para', 'com', 'é', 'e', 'que', 'um', 'uma',
            'gosta', 'prefere', 'precisa'
        }
        
        self.caracteristicas = {
            'luz': {'sol', 'luz', 'brilho', 'claro', 'sombra', 'escuro'},
            'agua': {'agua', 'molhado', 'seco', 'irrigar', 'regar'},
            'temperatura': {'quente', 'frio', 'temperatura', 'gelo', 'calor'},
            'umidade': {'umidade', 'seco', 'molhado', 'úmido'}
        }
    
    def limpar(self, texto):
        """Limpar texto"""
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto
    
    def extrair_caracteristicas(self, descricao):
        """Extrair necessidades da descrição"""
        texto_limpo = self.limpar(descricao)
        tokens = [t for t in texto_limpo.split() if t not in self.palavras_parada]
        
        caracteristicas_encontradas = {}
        for categoria, palavras in self.caracteristicas.items():
            encontradas = [t for t in tokens if t in palavras]
            if encontradas:
                caracteristicas_encontradas[categoria] = encontradas
        
        return caracteristicas_encontradas
    
    def gerar_recomendacoes(self, descricao):
        """Gerar recomendações baseadas em descrição"""
        caracteristicas = self.extrair_caracteristicas(descricao)
        
        recomendacoes = {
            'luz': 'moderada',
            'agua': 'moderada',
            'temp': '22°C',
            'umidade': 'média'
        }
        
        if 'luz' in caracteristicas:
            palavras = caracteristicas['luz']
            if any(p in ['sol', 'luz', 'brilho'] for p in palavras):
                recomendacoes['luz'] = 'alta'
            elif any(p in ['sombra', 'escuro'] for p in palavras):
                recomendacoes['luz'] = 'baixa'
        
        if 'agua' in caracteristicas:
            palavras = caracteristicas['agua']
            if any(p in ['molhado'] for p in palavras):
                recomendacoes['agua'] = 'alta'
            elif any(p in ['seco'] for p in palavras):
                recomendacoes['agua'] = 'baixa'
        
        return recomendacoes
```

#### 9.6 Sistema Principal (Orquestrador)

```python
# main.py
from dados import GerenciadorDados
from rede_neural import ModeloPlantas
from genetico import OtimizadorGenetico
from nlp import ProcessadorDescricoes

class SistemaRecomendacaoPlantas:
    def __init__(self):
        self.gerenciador_dados = GerenciadorDados()
        self.modelo = ModeloPlantas()
        self.processador_nlp = ProcessadorDescricoes()
        self.otimizador = None
    
    def inicializar(self):
        """Preparar tudo"""
        print("1. Carregando dados...")
        self.gerenciador_dados.carregar_dados_teste()
        
        print("2. Preparando dados...")
        X_treino, X_teste, y_treino, y_teste = self.gerenciador_dados.preparar_dados()
        
        print("3. Construindo modelo...")
        self.modelo.construir()
        
        print("4. Treinando modelo...")
        self.modelo.treinar(X_treino, y_treino, X_teste, y_teste)
        
        acuracia = self.modelo.avaliar(X_teste, y_teste)
        print(f"   Acurácia do modelo: {acuracia:.2%}")
        
        limites = {
            'luz': (4, 12),
            'agua': (100, 800),
            'temp': (16, 30),
            'umidade': (40, 75)
        }
        self.otimizador = OtimizadorGenetico(self.modelo, limites)
    
    def processar_planta(self, nome, descricao):
        """Processar planta e gerar recomendações"""
        print(f"\n{'='*60}")
        print(f"Analisando: {nome}")
        print(f"{'='*60}")
        
        print(f"\nDescrição: {descricao}")
        
        print("\n[NLP] Processando descrição...")
        recomendacoes_nlp = self.processador_nlp.gerar_recomendacoes(descricao)
        print(f"Características extraídas: {recomendacoes_nlp}")
        
        print("\n[Algoritmo Genético] Otimizando condições...")
        condicoes_otimas = self.otimizador.evoluir(gerações=15)
        
        print("\n[Rede Neural] Prevendo viabilidade...")
        import numpy as np
        vetor = np.array([
            condicoes_otimas['luz'],
            condicoes_otimas['agua'],
            condicoes_otimas['temp'],
            condicoes_otimas['umidade']
        ])
        
        probabilidade = self.modelo.prever(vetor)
        
        print("\n" + "="*60)
        print("RESULTADO FINAL")
        print("="*60)
        print(f"\nCondições Ideais Encontradas:")
        print(f"  • Luz solar: {condicoes_otimas['luz']:.1f} horas/dia")
        print(f"  • Água: {condicoes_otimas['agua']:.0f} ml/dia")
        print(f"  • Temperatura: {condicoes_otimas['temp']:.1f}°C")
        print(f"  • Umidade: {condicoes_otimas['umidade']:.1f}%")
        print(f"\nProbabilidade de Sobrevivência: {probabilidade*100:.1f}%")
        
        if probabilidade > 0.8:
            print("Recomendação: ✓ EXCELENTE - Planta tem alta viabilidade")
        elif probabilidade > 0.6:
            print("Recomendação: ◐ BOM - Planta viável com cuidados")
        else:
            print("Recomendação: ✗ DIFÍCIL - Requer muitos cuidados")
        
        return {
            'nome': nome,
            'condicoes': condicoes_otimas,
            'viabilidade': probabilidade
        }
    
    def executar_demo(self):
        """Executar demonstração completa"""
        print("\n" + "="*60)
        print("SISTEMA DE RECOMENDAÇÃO DE PLANTAS")
        print("Integrando: Rede Neural + Algoritmo Genético + NLP")
        print("="*60)
        
        self.inicializar()
        
        plantas = [
            {
                'nome': 'Orquídea',
                'descricao': 'Planta que prefere luz filtrada e ambiente úmido'
            },
            {
                'nome': 'Cacto',
                'descricao': 'Gosta de sol intenso e ambiente seco'
            },
            {
                'nome': 'Samambaia',
                'descricao': 'Prefere sombra e alta umidade'
            }
        ]
        
        resultados = []
        for planta in plantas:
            resultado = self.processar_planta(
                planta['nome'],
                planta['descricao']
            )
            resultados.append(resultado)
        
        print("\n" + "="*60)
        print("RESUMO GERAL")
        print("="*60)
        for r in resultados:
            status = "✓" if r['viabilidade'] > 0.6 else "✗"
            print(f"{status} {r['nome']}: {r['viabilidade']*100:.1f}% viável")

# Executar
if __name__ == "__main__":
    sistema = SistemaRecomendacaoPlantas()
    sistema.executar_demo()
```

---

### Capítulo 10: Como Tudo Se Conecta

#### 10.1 Fluxo de Dados Completo

```
ENTRADA (Usuário)
    ↓
[NLP] Processa descrição da planta
    ↓ (Extrai características)
[Rede Neural] Treina com dados históricos
    ↓ (Aprende padrões)
[Algoritmo Genético] Otimiza condições ideais
    ↓ (Testa muitas combinações)
SAÍDA (Condições ideais + Viabilidade)
```

#### 10.2 Explicação de Cada Componente

**1. NLP (Processamento de Linguagem Natural)**
   - Lê a descrição da planta em linguagem natural
   - Extrai palavras-chave relevantes
   - Converte em recomendações estruturadas
   - Traduz texto humano em dados processáveis

**2. Rede Neural**
   - Aprendeu padrões dos dados de treinamento
   - Prevê probabilidade de sobrevivência
   - Avalia se um conjunto de condições é bom
   - Age como "juiz" da qualidade das condições

**3. Algoritmo Genético**
   - Gera muitas possibilidades de condições
   - Testa cada uma com a Rede Neural
   - Mantém as melhores
   - Evolui iterativamente até encontrar condições ótimas
   - Funciona como "otimizador" de soluções

#### 10.3 Por Que Cada Tecnologia é Usada?

| Tecnologia | Função | Por Quê? |
|-----------|--------|---------|
| NLP | Entender descrição | Dados começam em texto livre |
| Rede Neural | Avaliar condições | Bom para padrões não-lineares |
| Algoritmo Genético | Encontrar ótimos | Bom para explorar grande espaço de soluções |

#### 10.4 Vantagens desta Arquitetura

- **Modular:** Cada componente funciona independentemente
- **Escalável:** Fácil adicionar novas plantas
- **Combina Forças:** NLP + NN + AG = solução robusta
- **Interpretável:** Cada etapa tem lógica clara

---

### Capítulo 11: Próximos Passos e Evolução

#### 11.1 Melhorias Possíveis

**Curto Prazo:**
- Adicionar mais plantas ao dataset
- Incluir mais características (luz UV, pH do solo)
- Implementar validação cruzada
- Adicionar interface gráfica

**Médio Prazo:**
- Integrar com banco de dados real
- Usar embeddings de palavra (Word2Vec)
- Implementar reinforcement learning
- Criar API REST

**Longo Prazo:**
- Usar Transformers (BERT, GPT)
- Integrar com sensores IoT reais
- Aplicar transfer learning
- Publicar como serviço web

#### 11.2 Adaptando para Outros Problemas

Essa arquitetura funciona para:
- **Recomendação de Filmes:** NLP para sinopse + NN para avaliação
- **Previsão de Demanda:** NN aprende padrões + AG otimiza estoque
- **Análise de Sentimento em Massa:** NLP processa + NN classifica
- **Design Automático:** AG gera designs + NN avalia

#### 11.3 Recursos para Aprofundamento

**Machine Learning:**
- Scikit-learn documentation
- Andrew Ng's Machine Learning Course

**Deep Learning:**
- TensorFlow/Keras Official Docs
- Fast.ai Courses

**NLP Avançado:**
- Hugging Face Transformers
- Stanford NLP Course

**Algoritmos Genéticos:**
- DEAP Library (Python)
- Papers sobre co-evolução

---

## PARTE 6: REFERÊNCIA RÁPIDA

### Capítulo 12: Guia de Troubleshooting

#### Problema: Modelo não aprende

**Causas possíveis:**
- Taxa de aprendizado muito alta/baixa
- Dados não normalizados
- Arquitetura inapropriada
- Overfitting

**Soluções:**
```python
# 1. Testar taxa de aprendizado
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# 2. Normalizar dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 3. Adicionar regularização
model.add(keras.layers.Dropout(0.3))

# 4. Validar com dados novos
X_treino, X_val, y_treino, y_val = train_test_split(X, y, test_size=0.2)
```

#### Problema: Algoritmo genético não converge

**Causas possíveis:**
- População muito pequena
- Taxa de mutação errada
- Critério de parada ruim

**Soluções:**
```python
# Aumentar população
ag = AlgoritmoGenetico(tamanho_populacao=200)

# Ajustar taxa de mutação
ag.taxa_mutacao = 0.05  # Começar em 5%

# Adicionar elitismo
melhores = sorted(populacao, key=fitness)[-10:]
nova_pop = melhores + nova_pop[10:]
```

#### Problema: NLP não identifica contexto

**Causas possíveis:**
- Vocabulário insuficiente
- Palavras-parada demais removidas
- Sem análise semântica

**Soluções:**
```python
# Usar dicionário maior
from nltk.corpus import stopwords
palavras_parada = set(stopwords.words('portuguese'))

# Usar embeddings
from gensim.models import Word2Vec
modelo_word2vec = Word2Vec(sentences, vector_size=100)

# Usar transformers
from transformers import pipeline
classif = pipeline("zero-shot-classification")
```

### Capítulo 13: Checklist de Implementação

Ao construir seu próprio sistema:

- [ ] Definir claramente o problema
- [ ] Coletar e explorar dados
- [ ] Normalizar dados
- [ ] Dividir em treino/teste
- [ ] Escolher arquitetura apropriada
- [ ] Implementar baseline simples
- [ ] Avaliar com métricas relevantes
- [ ] Iterar e otimizar
- [ ] Documentar código
- [ ] Testar em produção

### Capítulo 14: Glossário Rápido

**Acurácia:** Porcentagem de previsões corretas

**Backpropagation:** Algoritmo que ajusta pesos propagando erro para trás

**Batch:** Grupo de amostras processadas juntas

**Bias:** Parâmetro de ajuste em neurônios

**Convergência:** Quando modelo para de melhorar

**Dropout:** Técnica para evitar overfitting

**Época:** Uma passagem por todo dataset

**Fitness:** Qualidade de uma solução (algoritmos genéticos)

**Função de Ativação:** Função que adiciona não-linearidade

**Gradient:** Direção de máxima mudança

**Hiperparâmetro:** Configuração que você escolhe (não aprende)

**Loss:** Medida de erro do modelo

**Normalização:** Colocar dados em escala similar

**Overfitting:** Modelo memoriza dados em vez de generalizar

**Taxa de Aprendizado:** Quão rápido modelo aprende

**Tokenização:** Dividir texto em palavras

**Underfitting:** Modelo muito simples para capturar padrão

**Validação:** Testar em dados não-vistos

**Weight:** Parâmetro ajustado durante treinamento

---

## CONCLUSÃO

Você completou um caminho que vai desde conceitos fundamentais até implementar um sistema real integrando:

✓ **Rede Neural** - Para aprender padrões complexos
✓ **Algoritmo Genético** - Para otimizar soluções
✓ **NLP** - Para entender linguagem natural
✓ **Engenharia de Software** - Para tudo funcionar junto

### Seus Próximos Passos:

1. **Experimente:** Modifique o projeto com seus próprios dados
2. **Aprenda:** Estude cada componente mais profundamente
3. **Construa:** Crie seu próprio projeto do zero
4. **Compartilhe:** Mostre seus resultados

### Lembre-se:

- IA não é mágica, é matemática + dados
- Comece simples, evolua gradualmente
- Teste continuamente
- Documentação é essencial
- Comunidade Python é incrível

**Bem-vindo ao mundo da Inteligência Artificial!**
