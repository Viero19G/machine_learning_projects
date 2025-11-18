# CURSO COMPLETO: Domínio Unificado de Inteligência Artificial Moderna
## Do Zero ao Especialista em IA Integrada

---

# ÍNDICE GERAL

## MÓDULO 1: Fundamentos da IA (Capítulos 1-4)
## MÓDULO 2: Machine Learning (Capítulos 5-8)
## MÓDULO 3: Redes Neurais e Deep Learning (Capítulos 9-13)
## MÓDULO 4: Processamento de Linguagem Natural (Capítulos 14-17)
## MÓDULO 5: Algoritmos Genéticos e IA Evolutiva (Capítulos 18-19)
## MÓDULO 6: Visão Computacional (Capítulos 20-24)
## MÓDULO 7: Reinforcement Learning e RLHF (Capítulos 25-27)
## MÓDULO 8: Projeto Final Integrado (Capítulos 28-30)

---

# MÓDULO 1: FUNDAMENTOS DA IA

## Capítulo 1: O que é Inteligência Artificial?

### 1.1 Definição e Contexto Histórico

**Inteligência Artificial (IA)** é o desenvolvimento de máquinas capazes de executar tarefas que normalmente requerem inteligência humana. Essas tarefas incluem: aprender com experiência, reconhecer padrões, compreender linguagem, fazer decisões e resolver problemas.

```
1956 - Conferência de Dartmouth (nascimento oficial da IA)
1974 - Inverno da IA (expectativas não atendidas)
1980 - Retorno (sistemas especialistas)
2012 - Deep Learning revoluciona (AlexNet)
2017 - Era dos Transformers (BERT, GPT)
2023 - Era dos Modelos Grandes (ChatGPT, GPT-4)
```

### 1.2 Relacionamento entre IA, ML e DL

```
┌─────────────────────────────────────┐
│  INTELIGÊNCIA ARTIFICIAL (IA)       │
│  (Tudo que faz máquina parecer      │
│   inteligente)                       │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ MACHINE LEARNING (ML)        │  │
│  │ (Aprende através de dados)   │  │
│  │                              │  │
│  │ ┌────────────────────────┐   │  │
│  │ │ DEEP LEARNING (DL)     │   │  │
│  │ │ (Redes Neurais         │   │  │
│  │ │  profundas)            │   │  │
│  │ └────────────────────────┘   │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**IA Clássica:** Usa regras explícitas programadas
- Exemplo: Jogo de Xadrez (regras predefinidas)

**Machine Learning:** Aprende padrões dos dados
- Exemplo: Classificador de spam (aprende do dataset)

**Deep Learning:** Usa redes neurais complexas
- Exemplo: ChatGPT (bilhões de parâmetros)

### 1.3 Os Três Pilares Fundamentais

```python
# 1. REPRESENTAÇÃO: Como representamos o conhecimento?
dados = [1, 2, 3, 4, 5]  # Números
dados_texto = "gato"      # Strings
dados_imagem = [[0,1],[1,0]]  # Matrizes

# 2. APRENDIZADO: Como extraímos conhecimento dos dados?
def aprender(dados_historicos):
    padroes = encontrar_padroes(dados_historicos)
    return padroes

# 3. RACIOCÍNIO: Como usamos o conhecimento para tomar decisões?
def decidir(novo_dado, conhecimento):
    previsao = aplicar_conhecimento(novo_dado, conhecimento)
    return previsao
```

### 1.4 Tipos de Tarefas em IA

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Classificação** | Categorizar em classes | É spam ou não? |
| **Regressão** | Prever valores contínuos | Qual será o preço? |
| **Clustering** | Agrupar similaridades | Segmentar clientes |
| **Ranqueamento** | Ordenar por relevância | Resultados de busca |
| **Geração** | Criar novo conteúdo | Escrever texto, gerar imagem |
| **Tradução** | Converter entre domínios | Português → Inglês |
| **Otimização** | Encontrar melhor solução | Roteiro mais curto |

---

## Capítulo 2: IA Clássica - Conceitos Base

### 2.1 Lógica e Heurísticas

**Lógica Proposicional:** Verdadeiro ou Falso

```python
# IA Clássica: Sistema de Regras
class SistemaEspecialista:
    def __init__(self):
        self.regras = []
    
    def adicionar_regra(self, condicao, conclusao):
        """Adicionar regra SE-ENTÃO"""
        self.regras.append((condicao, conclusao))
    
    def inferir(self, fato):
        """Aplicar regras aos fatos"""
        for condicao, conclusao in self.regras:
            if condicao(fato):
                return conclusao
        return "Desconhecido"

# Exemplo
se = SistemaEspecialista()
se.adicionar_regra(
    lambda x: x > 30,
    "Pessoa é adulta"
)
se.adicionar_regra(
    lambda x: x <= 30,
    "Pessoa é jovem"
)

print(se.inferir(25))  # "Pessoa é jovem"
```

### 2.2 Busca e Algoritmos de Exploração

**Problema:** Encontrar caminho do ponto A ao B em um mapa

```python
import heapq

class AlgoritmoEstrela:
    """Algoritmo A* - Busca com heurística"""
    
    def __init__(self, mapa):
        self.mapa = mapa
    
    def heuristica(self, pos_atual, pos_alvo):
        """Distância Euclidiana"""
        return ((pos_atual[0] - pos_alvo[0])**2 + 
                (pos_atual[1] - pos_alvo[1])**2)**0.5
    
    def buscar_caminho(self, inicio, alvo):
        """Buscar caminho otimizado"""
        fila = [(0, inicio)]
        visitados = set()
        
        while fila:
            _, pos_atual = heapq.heappop(fila)
            
            if pos_atual in visitados:
                continue
            
            if pos_atual == alvo:
                return "Caminho encontrado!"
            
            visitados.add(pos_atual)
            
            # Explorar vizinhos
            for vizinho in self.obter_vizinhos(pos_atual):
                if vizinho not in visitados:
                    g = len(visitados)  # Distância percorrida
                    h = self.heuristica(vizinho, alvo)  # Estimativa
                    f = g + h  # Custo total
                    heapq.heappush(fila, (f, vizinho))
        
        return "Sem caminho"
    
    def obter_vizinhos(self, pos):
        """Retornar posições adjacentes válidas"""
        x, y = pos
        vizinhos = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        return [v for v in vizinhos if self.is_valido(v)]
    
    def is_valido(self, pos):
        """Verificar se posição é válida"""
        return 0 <= pos[0] < 10 and 0 <= pos[1] < 10
```

### 2.3 Classificadores Simples

```python
import numpy as np

class ClassificadorKNN:
    """K-Nearest Neighbors - Classificador simples e efetivo"""
    
    def __init__(self, k=3):
        self.k = k
        self.dados_treino = None
        self.labels_treino = None
    
    def treinar(self, dados, labels):
        """Armazenar dados de treino"""
        self.dados_treino = dados
        self.labels_treino = labels
    
    def distancia_euclidiana(self, ponto1, ponto2):
        """Calcular distância entre dois pontos"""
        return np.sqrt(np.sum((ponto1 - ponto2)**2))
    
    def classificar(self, ponto):
        """Classificar novo ponto"""
        # Calcular distância para todos os pontos de treino
        distancias = []
        for i, dado_treino in enumerate(self.dados_treino):
            dist = self.distancia_euclidiana(ponto, dado_treino)
            distancias.append((dist, self.labels_treino[i]))
        
        # Ordenar e pegar k mais próximos
        distancias.sort()
        vizinhos_proximos = distancias[:self.k]
        
        # Votação por maioria
        labels = [label for _, label in vizinhos_proximos]
        return max(set(labels), key=labels.count)

# Exemplo
dados = np.array([[1,2], [2,3], [3,1], [6,6], [7,7], [8,8]])
labels = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

knn = ClassificadorKNN(k=3)
knn.treinar(dados, labels)
print(knn.classificar(np.array([1.5, 2.5])))  # 'A'
```

---

## Capítulo 3: Fundamentos Matemáticos

### 3.1 Vetores e Matrizes

```python
import numpy as np

# VETOR: Lista de números (1D)
vetor = np.array([1, 2, 3, 4, 5])
print(f"Vetor: {vetor}")
print(f"Dimensão: {vetor.shape}")  # (5,)

# MATRIZ: Grade de números (2D)
matriz = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"Matriz:\n{matriz}")
print(f"Dimensão: {matriz.shape}")  # (3, 3)

# TENSOR: Estrutura multidimensional
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(f"Tensor dimensão: {tensor.shape}")  # (2, 2, 2)

# OPERAÇÕES BÁSICAS
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Soma: {a + b}")  # [5, 7, 9]
print(f"Produto elemento: {a * b}")  # [4, 10, 18]
print(f"Produto escalar: {np.dot(a, b)}")  # 32
print(f"Norma (magnitude): {np.linalg.norm(a)}")  # 3.74...
```

### 3.2 Funções e Derivadas

```python
import numpy as np
import matplotlib.pyplot as plt

# FUNÇÃO LINEAR
def linear(x):
    return 2*x + 1

# FUNÇÃO NÃO-LINEAR
def quadratica(x):
    return x**2

# FUNÇÃO EXPONENCIAL
def exponencial(x):
    return np.exp(x)

# DERIVADA (Variação)
def derivada_numerica(f, x, h=0.0001):
    """Aproximar derivada numericamente"""
    return (f(x + h) - f(x - h)) / (2 * h)

x = 3
print(f"f(x) = 2x+1 em x=3: {linear(x)}")
print(f"Derivada: {derivada_numerica(linear, x)}")  # ~2

print(f"f(x) = x² em x=3: {quadratica(x)}")
print(f"Derivada: {derivada_numerica(quadratica, x)}")  # ~6

# GRADIENTE DESCENDENTE (Otimização)
def gradiente_descendente(f_derivada, x_inicial, taxa_aprendizado=0.01, iteracoes=100):
    """Encontrar mínimo da função"""
    x = x_inicial
    historico = [x]
    
    for i in range(iteracoes):
        gradiente = f_derivada(x)
        x = x - taxa_aprendizado * gradiente
        historico.append(x)
    
    return x, historico

x_final, historico = gradiente_descendente(
    lambda x: 2*x,  # Derivada de x²
    x_inicial=5
)
print(f"Mínimo encontrado em x ≈ {x_final:.4f}")
```

### 3.3 Distribuições de Probabilidade

```python
import numpy as np
from scipy.stats import norm

# DISTRIBUIÇÃO NORMAL (Gaussiana)
media = 100
desvio_padrao = 15

# Gerar 1000 amostras
amostras = np.random.normal(media, desvio_padrao, 1000)

print(f"Média: {np.mean(amostras):.2f}")
print(f"Desvio padrão: {np.std(amostras):.2f}")

# PROBABILIDADE CONDICIONAL
# P(A|B) = P(A e B) / P(B)

class ProbabilidadeCondicional:
    def __init__(self):
        # Exemplo: Doença vs Teste
        self.casos_totais = 10000
        self.doentes = 100
        self.teste_positivo_doentes = 99  # Sensibilidade
        self.teste_positivo_saudaveis = 500  # Falso positivo
    
    def calcular_probabilidade(self):
        """P(tem doença | teste positivo)"""
        positivos_totais = self.teste_positivo_doentes + self.teste_positivo_saudaveis
        
        # Usando Bayes
        p_positivo_dado_doenca = self.teste_positivo_doentes / self.doentes
        p_doenca = self.doentes / self.casos_totais
        p_positivo = positivos_totais / self.casos_totais
        
        # P(D|T+) = P(T+|D) * P(D) / P(T+)
        p_doenca_dado_positivo = (p_positivo_dado_doenca * p_doenca) / p_positivo
        
        return p_doenca_dado_positivo

pc = ProbabilidadeCondicional()
prob = pc.calcular_probabilidade()
print(f"P(doença | teste positivo) = {prob:.2%}")  # ~16%
```

---

## Capítulo 4: Primeiros Algoritmos Práticos

### 4.1 Regressão Linear do Zero

```python
import numpy as np

class RegressaoLinearManual:
    """y = mx + b"""
    
    def __init__(self):
        self.m = 0
        self.b = 0
    
    def treinar(self, X, y, taxa_aprendizado=0.01, iteracoes=100):
        """Treinar usando gradiente descendente"""
        n = len(X)
        
        for _ in range(iteracoes):
            # Predições
            y_pred = self.m * X + self.b
            
            # Erros
            erros = y_pred - y
            
            # Gradientes
            grad_m = (2/n) * np.sum(X * erros)
            grad_b = (2/n) * np.sum(erros)
            
            # Atualizar parâmetros
            self.m -= taxa_aprendizado * grad_m
            self.b -= taxa_aprendizado * grad_b
    
    def prever(self, X):
        """Fazer predição"""
        return self.m * X + self.b

# Dados de exemplo
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # y ≈ 2x

modelo = RegressaoLinearManual()
modelo.treinar(X, y, taxa_aprendizado=0.01, iteracoes=100)

print(f"Coeficiente (m): {modelo.m:.4f}")  # ~2
print(f"Intercepto (b): {modelo.b:.4f}")   # ~0
print(f"Predição para x=6: {modelo.prever(6):.2f}")  # ~12
```

### 4.2 Clustering K-Means

```python
import numpy as np

class KMeans:
    """Agrupar dados em k clusters"""
    
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroides = None
    
    def inicializar_centroides(self, X):
        """Escolher k pontos aleatoriamente"""
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]
    
    def atribuir_cluster(self, X):
        """Encontrar cluster mais próximo para cada ponto"""
        distancias = np.zeros((len(X), self.k))
        
        for i in range(self.k):
            distancias[:, i] = np.sqrt(np.sum((X - self.centroides[i])**2, axis=1))
        
        return np.argmin(distancias, axis=1)
    
    def atualizar_centroides(self, X, clusters):
        """Recalcular centróides"""
        novos_centroides = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            pontos_cluster = X[clusters == i]
            if len(pontos_cluster) > 0:
                novos_centroides[i] = np.mean(pontos_cluster, axis=0)
        
        return novos_centroides
    
    def treinar(self, X):
        """Treinar modelo"""
        self.centroides = self.inicializar_centroides(X)
        
        for iteracao in range(self.max_iter):
            # Atribuir pontos
            clusters = self.atribuir_cluster(X)
            
            # Atualizar centróides
            novos_centroides = self.atualizar_centroides(X, clusters)
            
            # Verificar convergência
            if np.allclose(self.centroides, novos_centroides):
                break
            
            self.centroides = novos_centroides
        
        return clusters

# Dados de exemplo
X = np.array([
    [1, 1], [1.5, 2], [5, 8], [8, 8], [1, 0.5], [9, 9]
])

kmeans = KMeans(k=2)
clusters = kmeans.treinar(X)
print(f"Clusters: {clusters}")  # [0, 0, 1, 1, 0, 1]
```

---

# MÓDULO 2: MACHINE LEARNING

## Capítulo 5: Tipos de Aprendizado

### 5.1 Aprendizado Supervisionado

```python
import numpy as np

class AprendizadoSupervisionado:
    """
    Dados: (entrada, saída esperada)
    Objetivo: Aprender mapeamento entrada → saída
    """
    
    def __init__(self):
        self.exemplos_treino = []
    
    def adicionar_exemplo(self, entrada, saida_esperada):
        """Adicionar par (X, y)"""
        self.exemplos_treino.append((entrada, saida_esperada))
    
    def exemplo_classificacao(self):
        """Problema típico: Classificar email como spam ou não"""
        # Entrada: características do email
        # Saída: spam (1) ou não spam (0)
        
        exemplos = [
            ({"palavras": ["ganhar", "prêmio", "grátis"]}, 1),  # Spam
            ({"palavras": ["reunião", "amanhã", "projeto"]}, 0),  # Não spam
            ({"palavras": ["clique", "agora", "oferta"]}, 1),  # Spam
        ]
        return exemplos
    
    def exemplo_regressao(self):
        """Problema típico: Prever preço de imóvel"""
        # Entrada: características (área, quartos, etc.)
        # Saída: preço (valor contínuo)
        
        exemplos = [
            ({"area": 100, "quartos": 3}, 350000),
            ({"area": 150, "quartos": 4}, 500000),
            ({"area": 80, "quartos": 2}, 250000),
        ]
        return exemplos
```

### 5.2 Aprendizado Não Supervisionado

```python
class AprendizadoNaoSupervisionado:
    """
    Dados: apenas entrada (sem saída esperada)
    Objetivo: Descobrir estrutura nos dados
    """
    
    def exemplo_clustering(self):
        """Agrupar clientes similares sem rótulos"""
        # Sem saber "qual é a resposta certa"
        # O algoritmo descobre grupos naturais
        
        dados = [
            [25, 50000],  # Idade, renda
            [26, 55000],
            [55, 100000],
            [56, 105000],
        ]
        # Descobrir: 2 grupos (jovens e seniores)
        return dados
    
    def exemplo_reducao_dimensionalidade(self):
        """Reduzir 1000 features para 2-3 para visualizar"""
        # Muitas dimensões difíceis de entender
        # Reduzir mantendo informação essencial
        return "PCA, t-SNE"
    
    def exemplo_deteccao_anomalia(self):
        """Encontrar transações fraudulentas"""
        # Sem exemplos de fraude rotulados
        # Detectar comportamento atípico
        return "Isolation Forest, Autoencoders"
```

### 5.3 Aprendizado por Reforço

```python
class AprendizadoPorReforco:
    """
    Agente interage com ambiente
    Recebe recompensas/penalidades
    Aprende a maximizar recompensa total
    """
    
    def __init__(self):
        self.Q_table = {}  # Tabela de valores
        self.taxa_aprendizado = 0.1
        self.fator_desconto = 0.95
    
    def exemplo_videogame(self):
        """Agente aprende a jogar"""
        # Estado: posição, inimigos visíveis
        # Ações: mover, pular, atirar
        # Recompensa: vencer (+100), morrer (-50), +1 por ponto
        # Objetivo: maximizar score
        return "Algoritmo: Q-Learning, Policy Gradient"
    
    def exemplo_carro_autonomo(self):
        """Carro aprende a dirigir"""
        # Estado: velocidade, posição, sensores
        # Ações: acelerar, frear, virar
        # Recompensa: chegar no destino (+), colisão (-)
        # Objetivo: chegada segura
        return "Algoritmo: DQN, Actor-Critic"
    
    def atualizar_q_value(self, estado, acao, recompensa, proximo_estado):
        """Atualizar valor de ação-estado"""
        chave = (estado, acao)
        q_atual = self.Q_table.get(chave, 0)
        
        max_q_proximo = max(
            self.Q_table.get((proximo_estado, a), 0)
            for a in ['cima', 'baixo', 'esquerda', 'direita']
        )
        
        q_novo = q_atual + self.taxa_aprendizado * (
            recompensa + self.fator_desconto * max_q_proximo - q_atual
        )
        
        self.Q_table[chave] = q_novo
```

---

## Capítulo 6: Validação e Avaliação

### 6.1 Overfitting e Underfitting

```python
import numpy as np
import matplotlib.pyplot as plt

class ValidacaoModelo:
    """Entender quando modelo generaliza bem"""
    
    @staticmethod
    def exemplo_overfitting():
        """
        Modelo memoriza dados em vez de aprender padrão
        Péssimo em dados novos
        """
        print("""
        OVERFITTING:
        - Treino: Acurácia 99%
        - Teste: Acurácia 50%
        
        Causas:
        * Modelo muito complexo
        * Poucos dados de treino
        * Treinar por muitas épocas
        
        Soluções:
        * Regularização (L1, L2)
        * Early stopping
        * Aumentar dataset
        * Dropout (em redes neurais)
        """)
    
    @staticmethod
    def exemplo_underfitting():
        """
        Modelo muito simples, não captura padrão
        Ruim em treino E teste
        """
        print("""
        UNDERFITTING:
        - Treino: Acurácia 60%
        - Teste: Acurácia 58%
        
        Causas:
        * Modelo muito simples
        * Features insuficientes
        * Pouco treinamento
        
        Soluções:
        * Modelo mais complexo
        * Mais features
        * Treinar mais tempo
        """)
    
    @staticmethod
    def validacao_cruzada(dados, labels, k=5):
        """k-Fold Cross Validation"""
        tamanho_fold = len(dados) // k
        acuracias = []
        
        for i in range(k):
            inicio = i * tamanho_fold
            fim = inicio + tamanho_fold
            
            dados_teste = dados[inicio:fim]
            labels_teste = labels[inicio:fim]
            
            dados_treino = np.concatenate([dados[:inicio], dados[fim:]])
            labels_treino = np.concatenate([labels[:inicio], labels[fim:]])
            
            # Treinar e testar
            acuracia = np.random.random()  # Simulação
            acuracias.append(acuracia)
        
        return np.mean(acuracias)
```

### 6.2 Métricas de Avaliação

```python
class MetricasAvaliacao:
    """Medir performance do modelo"""
    
    @staticmethod
    def matriz_confusao(y_verdadeiro, y_predito):
        """
        TP (Verdadeiro Positivo): Predisse 1, era 1
        FP (Falso Positivo): Predisse 1, era 0
        TN (Verdadeiro Negativo): Predisse 0, era 0
        FN (Falso Negativo): Predisse 0, era 1
        """
        TP = sum((y_verdadeiro[i] == 1 and y_predito[i] == 1) for i in range(len(y_verdadeiro)))
        FP = sum((y_verdadeiro[i] == 0 and y_predito[i] == 1) for i in range(len(y_verdadeiro)))
        TN = sum((y_verdadeiro[i] == 0 and y_predito[i] == 0) for i in range(len(y_verdadeiro)))
        FN = sum((y_verdadeiro[i] == 1 and y_predito[i] == 0) for i in range(len(y_verdadeiro)))
        
        return TP, FP, TN, FN
    
    @staticmethod
    def calcular_metricas(TP, FP, TN, FN):
        """
        Acurácia: Proporção correta
        Precisão: De positivos preditos, quantos estavam certos?
        Recall: De positivos reais, quantos encontramos?
        F1-Score: Média harmônica (Precisão e Recall)
        """
        acuracia = (TP + TN) / (TP + FP + TN + FN)
        precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
        
        return {
            'acuracia': acuracia,
            'precisao': precisao,
            'recall': recall,
            'f1': f1
        }

# Exemplo
y_verdadeiro = [1, 0, 1, 1, 0, 1, 0, 0]
y_predito =    [1, 0, 1, 0, 0, 1, 1, 0]

TP, FP, TN, FN = MetricasAvaliacao.matriz_confusao(y_verdadeiro, y_predito)
metricas = MetricasAvaliacao.calcular_metricas(TP, FP, TN, FN)
print(metricas)
```

---

## Capítulo 7: Algoritmos de ML Principais

### 7.1 Árvores de Decisão

```python
class No:
    def __init__(self, feature=None, valor_divisao=None, esquerda=None, direita=None, classe=None):
        self.feature = feature
        self.valor_divisao = valor_divisao
        self.esquerda = esquerda
        self.direita = direita
        self.classe = classe  # Classe se nó folha

class ArvoreDecisao:
    """Classificador baseado em árvore"""
    
    def __init__(self, profundidade_max=5):
        self.raiz = None
        self.profundidade_max = profundidade_max
    
    def ganho_informacao(self, y_pai, y_esquerda, y_direita):
        """Calcular ganho de informação (redução de entropia)"""
        def entropia(y):
            p = np.mean(y) if len(y) > 0 else 0
            if p == 0 or p == 1:
                return 0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        n = len(y_pai)
        n_esq = len(y_esquerda)
        n_dir = len(y_direita)
        
        ent_pai = entropia(y_pai)
        ent_filha = (n_esq/n) * entropia(y_esquerda) + (n_dir/n) * entropia(y_direita)
        
        return ent_pai - ent_filha
    
    def melhor_divisao(self, X, y):
        """Encontrar melhor feature e valor para dividir"""
        melhor_ganho = -1
        melhor_feature = None
        melhor_valor = None
        
        for feature_idx in range(X.shape[1]):
            valores = np.unique(X[:, feature_idx])
            
            for valor in valores:
                mascara_esq = X[:, feature_idx] <= valor
                y_esq = y[mascara_esq]
                y_dir = y[~mascara_esq]
                
                if len(y_esq) == 0 or len(y_dir) == 0:
                    continue
                
                ganho = self.ganho_informacao(y, y_esq, y_dir)
                
                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_feature = feature_idx
                    melhor_valor = valor
        
        return melhor_feature, melhor_valor
    
    def construir_arvore(self, X, y, profundidade=0):
        """Construir árvore recursivamente"""
        # Critério de parada
        if len(np.unique(y)) == 1 or profundidade == self.profundidade_max or len(y) < 2:
            # Nó folha
            classe = np.bincount(y).argmax()
            return No(classe=classe)
        
        # Encontrar melhor divisão
        feature, valor = self.melhor_divisao(X, y)
        
        if feature is None:
            classe = np.bincount(y).argmax()
            return No(classe=classe)
        
        # Dividir dados
        mascara_esq = X[:, feature] <= valor
        
        # Construir subárvores
        no = No(feature=feature, valor_divisao=valor)
        no.esquerda = self.construir_arvore(X[mascara_esq], y[mascara_esq], profundidade+1)
        no.direita = self.construir_arvore(X[~mascara_esq], y[~mascara_esq], profundidade+1)
        
        return no
    
    def classificar_ponto(self, x, no):
        """Classificar um ponto"""
        if no.classe is not None:
            return no.classe
        
        if x[no.feature] <= no.valor_divisao:
            return self.classificar_ponto(x, no.esquerda)
        else:
            return self.classificar_ponto(x, no.direita)
    
    def treinar(self, X, y):
        """Treinar árvore"""
        self.raiz = self.construir_arvore(X, y)
    
    def prever(self, X):
        """Fazer predições"""
        return np.array([self.classificar_ponto(x, self.raiz) for x in X])
```

### 7.2 SVM (Support Vector Machine)

```python
class SVM:
    """Máquina de Vetores de Suporte - Classificador linear"""
    
    def __init__(self, taxa_aprendizado=0.01, iteracoes=1000, C=1.0):
        self.taxa_aprendizado = taxa_aprendizado
        self.iteracoes = iteracoes
        self.C = C  # Parâmetro de regularização
        self.w = None
        self.b = None
    
    def treinar(self, X, y):
        """Treinar SVM usando gradiente descendente"""
        n_amostras, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Garantir y = -1 ou 1
        y = np.where(y <= 0, -1, 1)
        
        for _ in range(self.iteracoes):
            for i in range(n_amostras):
                # Verificar margem
                decisao = y[i] * (np.dot(X[i], self.w) + self.b)
                
                if decisao < 1:
                    # Dentro da margem: atualizar
                    self.w -= self.taxa_aprendizado * (self.w - self.C * y[i] * X[i])
                    self.b -= self.taxa_aprendizado * (-self.C * y[i])
                else:
                    # Fora da margem: regularizar
                    self.w -= self.taxa_aprendizado * self.w
    
    def prever(self, X):
        """Fazer predições"""
        decisoes = np.dot(X, self.w) + self.b
        return np.sign(decisoes)
```

### 7.3 Ensemble: Random Forest

```python
class RandomForest:
    """Ensemble de múltiplas árvores"""
    
    def __init__(self, n_arvores=10):
        self.n_arvores = n_arvores
        self.arvores = []
    
    def bootstrap_amostra(self, X, y):
        """Gerar amostra com reposição"""
        n = len(X)
        indices = np.random.choice(n, n, replace=True)
        return X[indices], y[indices]
    
    def treinar(self, X, y):
        """Treinar múltiplas árvores"""
        for _ in range(self.n_arvores):
            X_amostra, y_amostra = self.bootstrap_amostra(X, y)
            arvore = ArvoreDecisao()
            arvore.treinar(X_amostra, y_amostra)
            self.arvores.append(arvore)
    
    def prever(self, X):
        """Votação por maioria"""
        predicoes = np.array([arvore.prever(X) for arvore in self.arvores])
        # Cada linha é uma árvore, cada coluna é uma amostra
        previsoes_finais = []
        for j in range(X.shape[0]):
            votos = predicoes[:, j]
            previsao_final = np.bincount(votos.astype(int)).argmax()
            previsoes_finais.append(previsao_final)
        return np.array(previsoes_finais)
```

---

## Capítulo 8: Projeto ML Integrado - Detector de Flores Iris

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ProjetoML_Iris:
    """Integrar todos os conceitos de ML em um projeto"""
    
    def __init__(self):
        self.X_treino = None
        self.X_teste = None
        self.y_treino = None
        self.y_teste = None
    
    def carregar_dados(self):
        """Carregar dataset Iris"""
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        # Dividir dados
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalizar
        scaler = StandardScaler()
        self.X_treino = scaler.fit_transform(self.X_treino)
        self.X_teste = scaler.transform(self.X_teste)
        
        print(f"[✓] Dataset carregado: {len(self.X_treino)} treino, {len(self.X_teste)} teste")
    
    def validacao_cruzada_comparativa(self):
        """Comparar múltiplos modelos"""
        modelos = {
            'KNN': ClassificadorKNN(k=3),
            'Árvore': ArvoreDecisao(),
            'Random Forest': RandomForest(n_arvores=5)
        }
        
        print("\n[Comparação de Modelos]")
        for nome, modelo in modelos.items():
            modelo.treinar(self.X_treino, self.y_treino)
            previsoes = modelo.prever(self.X_teste)
            acuracia = np.mean(previsoes == self.y_teste)
            print(f"  {nome}: {acuracia:.2%}")
    
    def executar(self):
        """Pipeline completo"""
        self.carregar_dados()
        self.validacao_cruzada_comparativa()

# Executar
projeto = ProjetoML_Iris()
projeto.executar()
```

---

# MÓDULO 3: REDES NEURAIS E DEEP LEARNING

## Capítulo 9: Perceptron e Neurônio Artificial

### 9.1 Neurônio Biológico vs Artificial

```python
import numpy as np

class Neuronio:
    """Neurônio artificial - unidade básica de rede neural"""
    
    def __init__(self, num_entradas):
        # Inicializar pesos e bias
        self.pesos = np.random.randn(num_entradas) * 0.01
        self.bias = 0
    
    def funcao_ativacao(self, z):
        """ReLU (Rectified Linear Unit)"""
        return np.maximum(0, z)
    
    def forward(self, entradas):
        """Forward pass: calcular saída"""
        z = np.dot(entradas, self.pesos) + self.bias
        return self.funcao_ativacao(z), z  # saída e z (para backprop)
    
    def backward(self, erro_entrada, z, taxa_aprendizado):
        """Backward pass: atualizar pesos"""
        # Derivada de ReLU
        dReLU = 1 if z > 0 else 0
        
        # Gradiente de erro
        delta = erro_entrada * dReLU
        
        # Atualizar pesos e bias
        self.pesos -= taxa_aprendizado * delta * self.pesos
        self.bias -= taxa_aprendizado * delta

class Perceptron:
    """Classificador linear simples (tipo especial de neurônio)"""
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def treinar(self, X, y, taxa_aprendizado=0.1, iteracoes=100):
        """Treinar perceptron"""
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(iteracoes):
            for i in range(len(X)):
                predicao = np.dot(X[i], self.w) + self.b
                predicao = 1 if predicao >= 0.5 else 0
                
                erro = y[i] - predicao
                
                # Atualizar pesos se houver erro
                if erro != 0:
                    self.w += taxa_aprendizado * erro * X[i]
                    self.b += taxa_aprendizado * erro
    
    def prever(self, X):
        """Fazer predição"""
        predicoes = np.dot(X, self.w) + self.b
        return (predicoes >= 0.5).astype(int)

# Exemplo: XOR (Perceptron não consegue!)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

perceptron = Perceptron()
perceptron.treinar(X_xor, y_xor)
print("Perceptron em XOR não é capaz de aprender")
```

---

## Capítulo 10: Rede Neural Feedforward

### 10.1 Arquitetura e Forward Pass

```python
class CamadaNeuralDensa:
    """Camada totalmente conectada"""
    
    def __init__(self, num_entradas, num_neuronios, funcao_ativacao='relu'):
        self.num_entradas = num_entradas
        self.num_neuronios = num_neuronios
        self.funcao_ativacao = funcao_ativacao
        
        # Inicializar pesos e bias
        self.W = np.random.randn(num_entradas, num_neuronios) * 0.01
        self.b = np.zeros((1, num_neuronios))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivada(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass"""
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        
        if self.funcao_ativacao == 'relu':
            self.A = self.relu(self.z)
        elif self.funcao_ativacao == 'sigmoid':
            self.A = self.sigmoid(self.z)
        elif self.funcao_ativacao == 'softmax':
            self.A = self.softmax(self.z)
        else:
            self.A = self.z
        
        return self.A
    
    def backward(self, dA, taxa_aprendizado):
        """Backward pass - Backpropagation"""
        m = self.X.shape[0]
        
        # Derivada da função de ativação
        if self.funcao_ativacao == 'relu':
            dZ = dA * self.relu_derivada(self.z)
        else:
            dZ = dA
        
        # Gradientes
        dW = np.dot(self.X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        
        # Atualizar pesos
        self.W -= taxa_aprendizado * dW
        self.b -= taxa_aprendizado * db
        
        return dX

class RedeNeural:
    """Rede Neural Feedforward completa"""
    
    def __init__(self, arquitetura):
        """
        arquitetura: lista com camadas
        Ex: [4, 8, 4, 1] = entrada 4, oculta 8, oculta 4, saída 1
        """
        self.camadas = []
        
        for i in range(len(arquitetura) - 1):
            if i == len(arquitetura) - 2:
                # Última camada (saída)
                funcao_ativacao = 'sigmoid'
            else:
                # Camadas ocultas
                funcao_ativacao = 'relu'
            
            camada = CamadaNeuralDensa(arquitetura[i], arquitetura[i+1], funcao_ativacao)
            self.camadas.append(camada)
    
    def forward(self, X):
        """Forward pass através de todas as camadas"""
        A = X
        for camada in self.camadas:
            A = camada.forward(A)
        return A
    
    def perda_entropia_cruzada(self, y_pred, y_real):
        """Calcular loss (erro)"""
        m = len(y_real)
        # Clipping para evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if y_real.ndim == 1:
            y_real = y_real.reshape(-1, 1)
        
        loss = -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_pred, y_real, taxa_aprendizado):
        """Backward pass através de todas as camadas"""
        m = len(y_real)
        if y_real.ndim == 1:
            y_real = y_real.reshape(-1, 1)
        
        # Gradiente da loss em relação à saída
        dA = y_pred - y_real
        
        # Backpropagate através das camadas
        for i in range(len(self.camadas) - 1, -1, -1):
            dA = self.camadas[i].backward(dA, taxa_aprendizado)
    
    def treinar(self, X, y, epocas=100, taxa_aprendizado=0.01, tamanho_batch=32):
        """Treinar rede neural"""
        historico_loss = []
        
        for epoca in range(epocas):
            # Embaralhar dados
            indices = np.random.permutation(len(X))
            X_embaralhado = X[indices]
            y_embaralhado = y[indices]
            
            # Mini-batch gradient descent
            loss_epoca = 0
            for i in range(0, len(X), tamanho_batch):
                X_batch = X_embaralhado[i:i+tamanho_batch]
                y_batch = y_embaralhado[i:i+tamanho_batch]
                
                # Forward
                y_pred = self.forward(X_batch)
                
                # Calcular loss
                loss = self.perda_entropia_cruzada(y_pred, y_batch)
                loss_epoca += loss
                
                # Backward
                self.backward(y_pred, y_batch, taxa_aprendizado)
            
            loss_media = loss_epoca / (len(X) // tamanho_batch)
            historico_loss.append(loss_media)
            
            if (epoca + 1) % 20 == 0:
                print(f"Época {epoca+1}/{epocas} | Loss: {loss_media:.4f}")
        
        return historico_loss
    
    def prever(self, X):
        """Fazer predição"""
        return self.forward(X)

# Exemplo: Resolver XOR com rede neural profunda
print("\n[Treinando Rede Neural para XOR]")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

rede = RedeNeural([2, 4, 4, 1])
historico = rede.treinar(X_xor, y_xor, epocas=100, taxa_aprendizado=0.5)

print("\nPredições XOR:")
for i in range(len(X_xor)):
    pred = rede.prever(X_xor[i:i+1])[0, 0]
    print(f"  {X_xor[i]} → {pred:.4f} (esperado: {y_xor[i, 0]})")
```

---

## Capítulo 11: Convolutional Neural Networks (CNN)

### 11.1 Operação de Convolução

```python
class CamadaConvolucional:
    """Camada de convolução para processar imagens"""
    
    def __init__(self, num_filtros, tamanho_filtro=3, stride=1, padding=0):
        self.num_filtros = num_filtros
        self.tamanho_filtro = tamanho_filtro
        self.stride = stride
        self.padding = padding
        
        # Inicializar filtros
        self.filtros = np.random.randn(num_filtros, tamanho_filtro, tamanho_filtro) * 0.01
    
    def convolver(self, entrada, filtro):
        """Aplicar um filtro de convolução"""
        resultado = 0
        for i in range(len(filtro)):
            for j in range(len(filtro[0])):
                resultado += entrada[i, j] * filtro[i, j]
        return resultado
    
    def forward(self, X):
        """Forward pass de convolução"""
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]
        
        batch_size, altura, largura = X.shape
        altura_saida = (altura - self.tamanho_filtro) // self.stride + 1
        largura_saida = (largura - self.tamanho_filtro) // self.stride + 1
        
        saida = np.zeros((batch_size, self.num_filtros, altura_saida, largura_saida))
        
        for b in range(batch_size):
            for f in range(self.num_filtros):
                for i in range(altura_saida):
                    for j in range(largura_saida):
                        # Extrair patch
                        i_inicio = i * self.stride
                        j_inicio = j * self.stride
                        patch = X[b, i_inicio:i_inicio+self.tamanho_filtro,
                                    j_inicio:j_inicio+self.tamanho_filtro]
                        
                        # Convolver
                        saida[b, f, i, j] = self.convolver(patch, self.filtros[f])
        
        return saida

class CamadaPooling:
    """Camada de pooling (redução de dimensionalidade)"""
    
    def __init__(self, tamanho_pool=2, tipo='max'):
        self.tamanho_pool = tamanho_pool
        self.tipo = tipo
    
    def forward(self, X):
        """Max pooling"""
        batch_size, canais, altura, largura = X.shape
        altura_saida = altura // self.tamanho_pool
        largura_saida = largura // self.tamanho_pool
        
        saida = np.zeros((batch_size, canais, altura_saida, largura_saida))
        
        for b in range(batch_size):
            for c in range(canais):
                for i in range(altura_saida):
                    for j in range(largura_saida):
                        i_inicio = i * self.tamanho_pool
                        j_inicio = j * self.tamanho_pool
                        patch = X[b, c, i_inicio:i_inicio+self.tamanho_pool,
                                     j_inicio:j_inicio+self.tamanho_pool]
                        
                        if self.tipo == 'max':
                            saida[b, c, i, j] = np.max(patch)
                        else:
                            saida[b, c, i, j] = np.mean(patch)
        
        return saida

# Exemplo: Processar imagem simples
print("\n[CNN - Processamento de Imagem]")
imagem = np.random.randn(28, 28)  # Imagem 28x28

conv = CamadaConvolucional(num_filtros=16, tamanho_filtro=3)
saida_conv = conv.forward(imagem)
print(f"Input: {imagem.shape}, Saída Conv: {saida_conv.shape}")

pooling = CamadaPooling(tamanho_pool=2)
saida_pool = pooling.forward(saida_conv)
print(f"Saída Pooling: {saida_pool.shape}")
```

---

## Capítulo 12: RNN e LSTM

### 12.1 Recurrent Neural Network

```python
class CamadaRNN:
    """Camada RNN básica para processar sequências"""
    
    def __init__(self, num_entradas, num_unidades_ocultas):
        self.num_entradas = num_entradas
        self.num_ocultas = num_unidades_ocultas
        
        # Pesos
        self.Wx = np.random.randn(num_entradas, num_unidades_ocultas) * 0.01
        self.Wh = np.random.randn(num_unidades_ocultas, num_unidades_ocultas) * 0.01
        self.b = np.zeros((1, num_unidades_ocultas))
    
    def forward(self, X):
        """Forward pass através de sequência"""
        tamanho_sequencia = len(X)
        h = np.zeros((1, self.num_ocultas))
        historico_h = [h]
        
        for t in range(tamanho_sequencia):
            # RNN: h_t = tanh(X_t * Wx + h_{t-1} * Wh + b)
            x_t = X[t:t+1]
            h = np.tanh(np.dot(x_t, self.Wx) + np.dot(h, self.Wh) + self.b)
            historico_h.append(h.copy())
        
        return h, historico_h

class LSTM:
    """Long Short-Term Memory - RNN avançada"""
    
    def __init__(self, num_entradas, num_unidades):
        self.num_entradas = num_entradas
        self.num_unidades = num_unidades
        
        # Pesos para as 4 portas
        self.Wf = np.random.randn(num_entradas + num_unidades, num_unidades) * 0.01
        self.Wi = np.random.randn(num_entradas + num_unidades, num_unidades) * 0.01
        self.Wo = np.random.randn(num_entradas + num_unidades, num_unidades) * 0.01
        self.Wc = np.random.randn(num_entradas + num_unidades, num_unidades) * 0.01
        
        # Biases
        self.bf = np.zeros((1, num_unidades))
        self.bi = np.zeros((1, num_unidades))
        self.bo = np.zeros((1, num_unidades))
        self.bc = np.zeros((1, num_unidades))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass LSTM"""
        tamanho_seq = len(X)
        h = np.zeros((1, self.num_unidades))
        c = np.zeros((1, self.num_unidades))  # Cell state
        
        for t in range(tamanho_seq):
            x_t = X[t:t+1]
            
            # Concatenar entrada e estado anterior
            concat = np.concatenate([x_t, h], axis=1)
            
            # Forget gate
            f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
            
            # Input gate
            i_t = self.sigmoid(np.dot(concat, self.Wi) + self.bi)
            
            # Candidate cell state
            c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Output gate
            o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
            
            # Update hidden state
            h = o_t * np.tanh(c)
        
        return h
```

---

## 13.1 Mecanismo de Atenção (Continuação)

```python
# Exemplo simplificado
print("\n[Mecanismo de Atenção]")
seq_len = 4
d_model = 8

Q = np.random.randn(seq_len, d_model)  # Queries
K = np.random.randn(seq_len, d_model)  # Keys
V = np.random.randn(seq_len, d_model)  # Values

saida, pesos_atencao = CalculoAtencao.atencao_escalonada(Q, K, V)
print(f"Saída atenção: {saida.shape}")
print(f"Pesos atenção:\n{pesos_atencao}")

class TransformerBlocoAtencao:
    """Bloco básico de Transformer"""
    
    def __init__(self, d_model=256, num_cabecas=8):
        self.d_model = d_model
        self.num_cabecas = num_cabecas
        self.d_k = d_model // num_cabecas
    
    def atencao_multi_cabeca(self, Q, K, V):
        """Multi-head attention"""
        saidas = []
        
        for i in range(self.num_cabecas):
            inicio = i * self.d_k
            fim = inicio + self.d_k
            
            Q_i = Q[:, inicio:fim]
            K_i = K[:, inicio:fim]
            V_i = V[:, inicio:fim]
            
            saida_i, _ = CalculoAtencao.atencao_escalonada(Q_i, K_i, V_i)
            saidas.append(saida_i)
        
        return np.concatenate(saidas, axis=1)

# Exemplo de multi-head attention
print("\n[Multi-Head Attention]")
transformer = TransformerBlocoAtencao(d_model=8, num_cabecas=2)
saida_multihead = transformer.atencao_multi_cabeca(Q, K, V)
print(f"Saída multi-head: {saida_multihead.shape}")
```

---


# MÓDULO 4: PROCESSAMENTO DE LINGUAGEM NATURAL (NLP)

## Capítulo 14: Fundamentos de NLP

### 14.1 Tokenização e Pré-processamento

```python
import re
from collections import Counter

class PreprocessadorTexto:
    """Preparar texto para análise"""
    
    def __init__(self):
        self.palavras_parada_pt = {
            'o', 'a', 'de', 'para', 'com', 'é', 'e', 'que', 'um', 'uma',
            'os', 'as', 'do', 'dos', 'da', 'das', 'em', 'ao', 'aos'
        }
    
    def limpar_texto(self, texto):
        """Limpeza básica"""
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto
    
    def tokenizar(self, texto):
        """Dividir em palavras"""
        texto_limpo = self.limpar_texto(texto)
        return texto_limpo.split()
    
    def remover_parada(self, tokens):
        """Remover palavras comuns"""
        return [t for t in tokens if t not in self.palavras_parada_pt and len(t) > 1]
    
    def processar(self, texto):
        """Pipeline completo"""
        tokens = self.tokenizar(texto)
        tokens = self.remover_parada(tokens)
        return tokens

# Exemplo
preprocessador = PreprocessadorTexto()
texto = "Inteligência Artificial é fascinante! Machine Learning revoluciona tudo."
tokens = preprocessador.processar(texto)
print(f"Tokens: {tokens}")
```

### 14.2 Bag of Words e TF-IDF

```python
import numpy as np

class BagOfWords:
    """Representação de texto como vetor de frequências"""
    
    def __init__(self):
        self.vocabulario = {}
        self.inverso_vocab = {}
    
    def construir_vocabulario(self, textos):
        """Criar mapeamento palavra → índice"""
        palavras_unicas = set()
        
        preprocessador = PreprocessadorTexto()
        for texto in textos:
            tokens = preprocessador.processar(texto)
            palavras_unicas.update(tokens)
        
        for i, palavra in enumerate(sorted(palavras_unicas)):
            self.vocabulario[palavra] = i
            self.inverso_vocab[i] = palavra
    
    def texto_para_vetor(self, texto):
        """Converter texto em vetor de frequências"""
        vetor = np.zeros(len(self.vocabulario))
        
        preprocessador = PreprocessadorTexto()
        tokens = preprocessador.processar(texto)
        
        for token in tokens:
            if token in self.vocabulario:
                idx = self.vocabulario[token]
                vetor[idx] += 1
        
        return vetor

class TfIdf:
    """TF-IDF: Term Frequency - Inverse Document Frequency"""
    
    def __init__(self):
        self.vocabulario = {}
        self.idf = {}
    
    def treinar(self, textos):
        """Calcular IDF"""
        preprocessador = PreprocessadorTexto()
        
        # Construir vocabulário
        palavras_unicas = set()
        for texto in textos:
            tokens = preprocessador.processar(texto)
            palavras_unicas.update(tokens)
        
        self.vocabulario = {p: i for i, p in enumerate(sorted(palavras_unicas))}
        
        # Calcular IDF
        num_docs = len(textos)
        for palavra in self.vocabulario:
            docs_com_palavra = sum(1 for texto in textos 
                                  if palavra in preprocessador.processar(texto))
            self.idf[palavra] = np.log(num_docs / (1 + docs_com_palavra))
    
    def texto_para_vetor(self, texto):
        """Converter para TF-IDF"""
        vetor = np.zeros(len(self.vocabulario))
        
        preprocessador = PreprocessadorTexto()
        tokens = preprocessador.processar(texto)
        
        # TF: frequência do termo
        for token in tokens:
            if token in self.vocabulario:
                idx = self.vocabulario[token]
                vetor[idx] += 1
        
        # Normalizar TF
        if np.sum(vetor) > 0:
            vetor = vetor / np.sum(vetor)
        
        # Multiplicar por IDF
        for palavra, idx in self.vocabulario.items():
            vetor[idx] *= self.idf.get(palavra, 0)
        
        return vetor

# Exemplo
textos = [
    "Inteligência Artificial é fascinante",
    "Machine Learning revoluciona tecnologia",
    "Python é ótimo para IA"
]

tfidf = TfIdf()
tfidf.treinar(textos)

vetor1 = tfidf.texto_para_vetor("Inteligência Artificial")
print(f"Vetor TF-IDF: {vetor1}")
```

### 14.3 Word Embeddings (Word2Vec)

```python
class Word2Vec:
    """Word embeddings - representar palavras em espaço contínuo"""
    
    def __init__(self, tamanho_embedding=100, janela=2):
        self.tamanho_embedding = tamanho_embedding
        self.janela = janela
        self.vocabulario = {}
        self.vetores_palavra = {}
    
    def treinar(self, textos, iteracoes=100, taxa_aprendizado=0.01):
        """Treinar embeddings usando Skip-gram"""
        preprocessador = PreprocessadorTexto()
        
        # Construir vocabulário
        palavras = []
        for texto in textos:
            tokens = preprocessador.processar(texto)
            palavras.extend(tokens)
        
        palavras_unicas = list(set(palavras))
        self.vocabulario = {p: i for i, p in enumerate(palavras_unicas)}
        
        # Inicializar vetores aleatoriamente
        for palavra in palavras_unicas:
            self.vetores_palavra[palavra] = np.random.randn(self.tamanho_embedding) * 0.01
        
        # Treinar Skip-gram
        for _ in range(iteracoes):
            for i, palavra_alvo in enumerate(palavras):
                # Contexto
                inicio = max(0, i - self.janela)
                fim = min(len(palavras), i + self.janela + 1)
                
                for j in range(inicio, fim):
                    if i != j:
                        palavra_contexto = palavras[j]
                        
                        # Predição simples
                        pred = np.dot(self.vetores_palavra[palavra_alvo],
                                    self.vetores_palavra[palavra_contexto])
                        
                        # Atualizar (simulado)
                        erro = (1 - pred)
                        self.vetores_palavra[palavra_alvo] += taxa_aprendizado * erro * self.vetores_palavra[palavra_contexto]
    
    def similaridade(self, palavra1, palavra2):
        """Calcular similaridade entre duas palavras"""
        if palavra1 not in self.vetores_palavra or palavra2 not in self.vetores_palavra:
            return 0
        
        v1 = self.vetores_palavra[palavra1]
        v2 = self.vetores_palavra[palavra2]
        
        norma1 = np.linalg.norm(v1)
        norma2 = np.linalg.norm(v2)
        
        if norma1 == 0 or norma2 == 0:
            return 0
        
        return np.dot(v1, v2) / (norma1 * norma2)
```

---

## Capítulo 15: Classificação de Texto

```python
class ClassificadorTexto:
    """Classificar textos em categorias"""
    
    def __init__(self):
        self.tfidf = TfIdf()
        self.modelo = RandomForest(n_arvores=10)
    
    def treinar(self, textos, labels):
        """Treinar classificador"""
        self.tfidf.treinar(textos)
        
        # Converter textos para vetores
        X = np.array([self.tfidf.texto_para_vetor(t) for t in textos])
        y = np.array(labels)
        
        # Treinar modelo
        self.modelo.treinar(X, y)
    
    def classificar(self, texto):
        """Classificar novo texto"""
        vetor = self.tfidf.texto_para_vetor(texto).reshape(1, -1)
        return self.modelo.prever(vetor)[0]

# Exemplo: Detector de Sentimento
textos_treino = [
    "Adorei o filme, foi excelente!",
    "Produto maravilhoso, muito feliz",
    "Que coisa horrível, muito ruim",
    "Decepcionante, não recomendo"
]

labels = [1, 1, 0, 0]  # 1 = positivo, 0 = negativo

classificador = ClassificadorTexto()
classificador.treinar(textos_treino, labels)

print("Classificação de sentimento:")
print(classificador.classificar("Adorei tudo!"))  # 1 (positivo)
print(classificador.classificar("Muito ruim"))    # 0 (negativo)
```

---

## Capítulo 16: Construindo um Chatbot Simples

```python
class ChatbotSimples:
    """Chatbot baseado em patterns e respostas"""
    
    def __init__(self):
        self.base_conhecimento = {}
    
    def adicionar_conhecimento(self, pergunta, resposta):
        """Adicionar par pergunta-resposta"""
        preprocessador = PreprocessadorTexto()
        pergunta_processada = ' '.join(preprocessador.processar(pergunta))
        self.base_conhecimento[pergunta_processada] = resposta
    
    def encontrar_melhor_correspondencia(self, pergunta):
        """Encontrar resposta mais similar"""
        preprocessador = PreprocessadorTexto()
        pergunta_processada = ' '.join(preprocessador.processar(pergunta))
        
        melhor_similaridade = 0
        melhor_resposta = "Desculpe, não entendi"
        
        palavras_pergunta = set(pergunta_processada.split())
        
        for chave, resposta in self.base_conhecimento.items():
            palavras_chave = set(chave.split())
            
            # Similaridade Jaccard
            intersecao = len(palavras_pergunta & palavras_chave)
            uniao = len(palavras_pergunta | palavras_chave)
            similaridade = intersecao / uniao if uniao > 0 else 0
            
            if similaridade > melhor_similaridade:
                melhor_similaridade = similaridade
                melhor_resposta = resposta
        
        return melhor_resposta
    
    def responder(self, pergunta):
        """Responder pergunta"""
        return self.encontrar_melhor_correspondencia(pergunta)

# Exemplo
chatbot = ChatbotSimples()
chatbot.adicionar_conhecimento("Qual é seu nome?", "Sou um chatbot simples")
chatbot.adicionar_conhecimento("Como você está?", "Estou funcionando normalmente")
chatbot.adicionar_conhecimento("O que é IA?", "IA é Inteligência Artificial")

print("Bot:", chatbot.responder("Qual seu nome?"))
print("Bot:", chatbot.responder("Como vai você?"))
```

---

# MÓDULO 5: ALGORITMOS GENÉTICOS E IA EVOLUTIVA

## Capítulo 18: Fundamentos de Algoritmos Genéticos

```python
class AlgoritmoGenetico:
    """Otimização através de evolução simulada"""
    
    def __init__(self, tamanho_populacao=50, taxa_mutacao=0.1):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.populacao = []
        self.melhor_fitness_historico = []
    
    def criar_populacao_aleatoria(self, tamanho_cromossomo):
        """Gerar população inicial"""
        self.populacao = [
            [np.random.randint(0, 2) for _ in range(tamanho_cromossomo)]
            for _ in range(self.tamanho_populacao)
        ]
    
    def calcular_fitness(self, individuo, funcao_fitness):
        """Avaliar qualidade do indivíduo"""
        return funcao_fitness(individuo)
    
    def selecionar_pais(self, populacao, fitness_scores):
        """Selecionar melhores indivíduos"""
        indices = np.argsort(fitness_scores)[-self.tamanho_populacao//2:]
        return [populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        """Combinar dois pais"""
        ponto_corte = len(pai1) // 2
        filho = pai1[:ponto_corte] + pai2[ponto_corte:]
        return filho
    
    def mutar(self, individuo):
        """Aplicar mutação aleatória"""
        for i in range(len(individuo)):
            if np.random.random() < self.taxa_mutacao:
                individuo[i] = 1 - individuo[i]
        return individuo
    
    def evoluir(self, funcao_fitness, gerações=100):
        """Executar algoritmo genético"""
        tamanho_cromossomo = len(self.populacao[0]) if self.populacao else 20
        self.criar_populacao_aleatoria(tamanho_cromossomo)
        
        for geracao in range(gerações):
            # Calcular fitness
            fitness_scores = [
                self.calcular_fitness(ind, funcao_fitness)
                for ind in self.populacao
            ]
            
            melhor_fitness = max(fitness_scores)
            self.melhor_fitness_historico.append(melhor_fitness)
            
            # Selecionar pais
            pais = self.selecionar_pais(self.populacao, fitness_scores)
            
            # Criar nova população
            nova_populacao = []
            while len(nova_populacao) < self.tamanho_populacao:
                pai1, pai2 = np.random.choice(len(pais), 2, replace=False)
                filho = self.crossover(pais[pai1], pais[pai2])
                filho = self.mutar(filho)
                nova_populacao.append(filho)
            
            self.populacao = nova_populacao
            
            if (geracao + 1) % 20 == 0:
                print(f"Geração {geracao+1}: Melhor fitness = {melhor_fitness:.4f}")
        
        # Retornar melhor indivíduo
        fitness_scores = [
            self.calcular_fitness(ind, funcao_fitness)
            for ind in self.populacao
        ]
        melhor_idx = np.argmax(fitness_scores)
        return self.populacao[melhor_idx]

# Exemplo: Maximizar função
def funcao_objetivo(cromossomo):
    """Converter cromossomo em número e calcular fitness"""
    numero = sum(bit * (2**i) for i, bit in enumerate(cromossomo))
    return numero

ag = AlgoritmoGenetico()
ag.criar_populacao_aleatoria(8)
melhor = ag.evoluir(funcao_objetivo, gerações=50)
print(f"Melhor solução: {melhor}")
```

## Capítulo 19: Problema do Caixeiro Viajante com AG

```python
class TSPGenetico:
    """Resolver TSP com Algoritmo Genético"""
    
    def __init__(self, cidades, tamanho_pop=50):
        self.cidades = np.array(cidades)
        self.tamanho_pop = tamanho_pop
        self.num_cidades = len(cidades)
        self.populacao = self.criar_populacao()
    
    def criar_populacao(self):
        """Gerar rotas aleatórias"""
        pop = []
        for _ in range(self.tamanho_pop):
            rota = list(np.random.permutation(self.num_cidades))
            pop.append(rota)
        return pop
    
    def calcular_distancia(self, rota):
        """Calcular distância total da rota"""
        distancia = 0
        for i in range(len(rota)):
            cidade_atual = rota[i]
            cidade_proxima = rota[(i + 1) % len(rota)]
            dist = np.linalg.norm(
                self.cidades[cidade_atual] - self.cidades[cidade_proxima]
            )
            distancia += dist
        return distancia
    
    def selecionar(self):
        """Selecionar melhores rotas"""
        distancias = [self.calcular_distancia(rota) for rota in self.populacao]
        indices = np.argsort(distancias)[:self.tamanho_pop//2]
        return [self.populacao[i] for i in indices]
    
    def crossover(self, pai1, pai2):
        """Crossover em ordem preservada"""
        tamanho = len(pai1)
        ponto1, ponto2 = sorted(np.random.choice(tamanho, 2, replace=False))
        
        filho = [-1] * tamanho
        filho[ponto1:ponto2] = pai1[ponto1:ponto2]
        
        idx = ponto2
        for gene in pai2:
            if gene not in filho:
                filho[idx % tamanho] = gene
                idx += 1
        
        return filho
    
    def mutar(self, rota):
        """Trocar duas cidades aleatoriamente"""
        if np.random.random() < 0.1:
            i, j = np.random.choice(len(rota), 2, replace=False)
            rota[i], rota[j] = rota[j], rota[i]
        return rota
    
    def evoluir(self, gerações=100):
        """Executar AG para TSP"""
        for gen in range(gerações):
            pais = self.selecionar()
            nova_pop = []
            
            while len(nova_pop) < self.tamanho_pop:
                pai1, pai2 = np.random.choice(len(pais), 2, replace=False)
                filho = self.crossover(pais[pai1], pais[pai2])
                filho = self.mutar(filho)
                nova_pop.append(filho)
            
            self.populacao = nova_pop
            
            if (gen + 1) % 20 == 0:
                melhor_dist = min(self.calcular_distancia(r) for r in self.populacao)
                print(f"Geração {gen+1}: Melhor distância = {melhor_dist:.2f}")
        
        # Retornar melhor rota
        melhor_rota = min(self.populacao, key=self.calcular_distancia)
        return melhor_rota, self.calcular_distancia(melhor_rota)

# Exemplo
cidades = [(0, 0), (10, 5), (15, 10), (5, 15), (2, 8)]
tsp = TSPGenetico(cidades)
rota, distancia = tsp.evoluir(gerações=50)
print(f"Melhor rota: {rota}, Distância: {distancia:.2f}")
```

---

# MÓDULO 6: VISÃO COMPUTACIONAL

## Capítulo 20: Manipulação de Imagens Básica

```python
import cv2
import numpy as np
from PIL import Image

class ProcessadorImagem:
    """Operações básicas com imagens"""
    
    @staticmethod
    def carregar_imagem(caminho):
        """Carregar imagem"""
        imagem = cv2.imread(caminho)
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def escala_cinza(imagem):
        """Converter para escala de cinza"""
        return cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    @staticmethod
    def redimensionar(imagem, largura, altura):
        """Redimensionar imagem"""
        return cv2.resize(imagem, (largura, altura))
    
    @staticmethod
    def borrão(imagem, kernel_size=5):
        """Aplicar borrão Gaussiano"""
        return cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def deteccao_bordas(imagem):
        """Detectar bordas com Canny"""
        cinza = ProcessadorImagem.escala_cinza(imagem)
        return cv2.Canny(cinza, 100, 200)
    
    @staticmethod
    def limiarizar(imagem, limiar=127):
        """Aplicar limiarização"""
        cinza = ProcessadorImagem.escala_cinza(imagem)
        _, imagem_limiar = cv2.threshold(cinza, limiar, 255, cv2.THRESH_BINARY)
        return imagem_limiar

# Exemplo
print("[Processamento de Imagem]")
# imagem = ProcessadorImagem.carregar_imagem("imagem.jpg")
# imagem_cinza = ProcessadorImagem.escala_cinza(imagem)
# bordas = ProcessadorImagem.deteccao_bordas(imagem)
```

## Capítulo 21: Detecção em Tempo Real com Webcam

```python
class DetectorWebcam:
    """Capturar e processar video em tempo real"""
    
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
    
    def capturar_frame(self):
        """Capturar um frame"""
        ret, frame = self.camera.read()
        return frame if ret else None
    
    def processar_frame(self, frame):
        """Processar frame"""
        # Redimensionar
        frame = cv2.resize(frame, (640, 480))
        
        # Detectar bordas
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bordas = cv2.Canny(cinza, 100, 200)
        
        return frame, bordas
    
    def exibir_video(self, duracao=10):
        """Exibir video em tempo real"""
        print(f"Capturando por {duracao} segundos... Pressione Q para parar")
        
        tempo_inicio = cv2.getTickCount()
        
        while True:
            frame = self.capturar_frame()
            if frame is None:
                break
            
            frame_processado, bordas = self.processar_frame(frame)
            
            # Mostrar
            cv2.imshow("Video", frame_processado)
            cv2.imshow("Bordas", bordas)
            
            # Sair com Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Verificar tempo limite
            tempo_decorrido = (cv2.getTickCount() - tempo_inicio) / cv2.getTickFrequency()
            if tempo_decorrido > duracao:
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
```

## Capítulo 22: Detecção de Objetos Clássica (Haar Cascades)

```python
class DetectorFaces:
    """Detectar rostos usando Haar Cascades"""
    
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detectar(self, imagem):
        """Detectar rostos"""
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        rostos = self.cascade.detectMultiScale(cinza, 1.3, 5)
        return rostos
    
    def desenhar_deteccoes(self, imagem, rostos):
        """Desenhar retângulos ao redor dos rostos"""
        for (x, y, w, h) in rostos:
            cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return imagem
    
    def processar_video(self):
        """Processar video com detecção de rostos"""
        camera = cv2.VideoCapture(0)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            rostos = self.detectar(frame)
            frame_desenhado = self.desenhar_deteccoes(frame, rostos)
            
            cv2.imshow("Deteccao de Rostos", frame_desenhado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()
```

## Capítulo 23: Detecção Moderna com YOLO

```python
class DetectorYOLO:
    """YOLO - You Only Look Once (Modern Object Detection)"""
    
    def __init__(self, modelo_peso='yolov3.weights', config='yolov3.cfg'):
        self.net = cv2.dnn.readNetFromDarknet(config, modelo_weight)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] 
                             for i in self.net.getUnconnectedOutLayers()]
        
        # Classes COCO
        self.classes = ["person", "car", "dog", "cat", "bird"]  # Subset
    
    def detectar(self, imagem, conf_threshold=0.5):
        """Detectar objetos"""
        altura, largura = imagem.shape[:2]
        
        # Preparar blob
        blob = cv2.dnn.blobFromImage(imagem, 0.00392, (416, 416), (0, 0, 0), True, False)
        self.net.setInput(blob)
        
        # Prever
        outputs = self.net.forward(self.output_layers)
        
        deteccoes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    center_x = int(detection[0] * largura)
                    center_y = int(detection[1] * altura)
                    w = int(detection[2] * largura)
                    h = int(detection[3] * altura)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    deteccoes.append((x, y, w, h, class_id, confidence))
        
        return deteccoes
```

## Capítulo 24: Reconhecimento Facial e OCR

```python
class ReconhecedorFacial:
    """Reconhecer identidade de pessoas"""
    
    def __init__(self):
        self.detector = cv2.face.LBPHFaceRecognizer_create()
        self.faces_known = []
        self.labels_known = []
    
    def treinar(self, imagens, labels):
        """Treinar reconhecedor"""
        self.faces_known = imagens
        self.labels_known = labels
        
        # Converter para escala de cinza
        imagens_cinza = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imagens]
        
        self.detector.train(imagens_cinza, np.array(labels))
    
    def reconhecer(self, imagem):
        """Reconhecer rosto"""
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        label, confianca = self.detector.predict(cinza)
        return label, confianca

class ReconhecedorOCR:
    """Optical Character Recognition - Reconhecer texto em imagens"""
    
    @staticmethod
    def extrair_texto_tesseract(imagem_path):
        """Usar Tesseract para OCR"""
        # Requer: pip install pytesseract
        import pytesseract
        imagem = cv2.imread(imagem_path)
        texto = pytesseract.image_to_string(imagem)
        return texto
    
    @staticmethod
    def extrair_texto_easyocr(imagem_path):
        """Usar EasyOCR"""
        # Requer: pip install easyocr
        import easyocr
        reader = easyocr.Reader(['pt'])
        resultado = reader.readtext(imagem_path)
        return resultado
```

---

# MÓDULO 7: REINFORCEMENT LEARNING E RLHF

## Capítulo 25: Q-Learning Básico

```python
class QLearning:
    """Aprender política ótima através de Q-Learning"""
    
    def __init__(self, num_estados, num_acoes, taxa_aprendizado=0.1, fator_desconto=0.95):
        self.num_estados = num_estados
        self.num_acoes = num_acoes
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_desconto = fator_desconto
        
        # Tabela Q: (estado, ação) → valor
        self.Q = np.zeros((num_estados, num_acoes))
    
    def selecionar_acao(self, estado, epsilon=0.1):
        """Exploration vs Exploitation (Epsilon-Greedy)"""
        if np.random.random() < epsilon:
            # Explorar: ação aleatória
            return np.random.randint(self.num_acoes)
        else:
            # Explorar: melhor ação conhecida
            return np.argmax(self.Q[estado, :])
    
    def atualizar(self, estado, acao, recompensa, proximo_estado):
        """Atualizar Q-value"""
        # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
        q_antigo = self.Q[estado, acao]
        q_maximo_proximo = np.max(self.Q[proximo_estado, :])
        
        q_novo = q_antigo + self.taxa_aprendizado * (
            recompensa + self.fator_desconto * q_maximo_proximo - q_antigo
        )
        
        self.Q[estado, acao] = q_novo
    
    def treinar(self, ambiente, num_episodios=100):
        """Treinar usando Q-Learning"""
        for episodio in range(num_episodios):
            estado = 0  # Estado inicial
            recompensa_total = 0
            
            for _ in range(100):  # Max passos por episódio
                # Selecionar ação
                acao = self.selecionar_acao(estado, epsilon=0.1)
                
                # Executar ação
                proximo_estado, recompensa, fim = ambiente(estado, acao)
                
                # Atualizar Q-value
                self.atualizar(estado, acao, recompensa, proximo_estado)
                
                recompensa_total += recompensa
                estado = proximo_estado
                
                if fim:
                    break
            
            if (episodio + 1) % 20 == 0:
                print(f"Episódio {episodio+1}: Recompensa = {recompensa_total}")

# Exemplo: Gridworld simples
class GridWorld:
    """Ambiente simples para teste"""
    
    def __init__(self, tamanho=5):
        self.tamanho = tamanho
        self.posicao = 0
        self.objetivo = tamanho * tamanho - 1
    
    def step(self, acao):
        """Executar ação e retornar novo estado e recompensa"""
        # Ações: 0=cima, 1=direita, 2=baixo, 3=esquerda
        deslocamentos = {0: -self.tamanho, 1: 1, 2: self.tamanho, 3: -1}
        
        nova_posicao = self.posicao + deslocamentos.get(acao, 0)
        nova_posicao = max(0, min(self.tamanho * self.tamanho - 1, nova_posicao))
        
        self.posicao = nova_posicao
        
        recompensa = 1 if nova_posicao == self.objetivo else -0.1
        fim = nova_posicao == self.objetivo
        
        return nova_posicao, recompensa, fim
```

## Capítulo 26: Policy Gradient e Actor-Critic

```python
class PolicyGradient:
    """Aprender política diretamente (sem Q-values)"""
    
    def __init__(self, num_estados, num_acoes, taxa_aprendizado=0.01):
        self.num_estados = num_estados
        self.num_acoes = num_acoes
        self.taxa_aprendizado = taxa_aprendizado
        
        # Pesos da política
        self.W = np.random.randn(num_estados, num_acoes) * 0.01
    
    def obter_probabilidades_acoes(self, estado):
        """Calcular probabilidade de cada ação (softmax)"""
        logits = self.W[estado, :]
        exp_logits = np.exp(logits - np.max(logits))
        probabilidades = exp_logits / np.sum(exp_logits)
        return probabilidades
    
    def selecionar_acao(self, estado):
        """Selecionar ação baseado em probabilidades"""
        probs = self.obter_probabilidades_acoes(estado)
        acao = np.random.choice(self.num_acoes, p=probs)
        return acao, probs[acao]
    
    def atualizar_politica(self, estado, acao, recompensa):
        """Atualizar pesos usando policy gradient"""
        probs = self.obter_probabilidades_acoes(estado)
        
        # Gradiente: ∇log(π(a|s))
        grad = np.zeros(self.num_acoes)
        grad[acao] = 1
        grad -= probs
        
        # Atualizar pesos
        self.W[estado, :] += self.taxa_aprendizado * recompensa * grad

class ActorCritic:
    """Combina Policy Gradient (Actor) e Value Function (Critic)"""
    
    def __init__(self, num_estados, num_acoes):
        self.num_estados = num_estados
        self.num_acoes = num_acoes
        
        # Actor: política
        self.W_actor = np.random.randn(num_estados, num_acoes) * 0.01
        
        # Critic: função de valor
        self.W_critic = np.random.randn(num_estados, 1) * 0.01
    
    def obter_valor(self, estado):
        """Estimar valor de estado"""
        return self.W_critic[estado, 0]
    
    def selecionar_acao(self, estado):
        """Selecionar ação"""
        logits = self.W_actor[estado, :]
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        acao = np.random.choice(self.num_acoes, p=probs)
        return acao
    
    def atualizar(self, estado, acao, recompensa, proximo_estado, gamma=0.95):
        """Atualizar ambos actor e critic"""
        # TD error: vantagem
        v_atual = self.obter_valor(estado)
        v_proximo = self.obter_valor(proximo_estado)
        td_error = recompensa + gamma * v_proximo - v_atual
        
        # Atualizar Critic
        self.W_critic[estado, 0] += 0.01 * td_error
        
        # Atualizar Actor
        probs = np.exp(self.W_actor[estado, :])
        probs = probs / np.sum(probs)
        grad = np.zeros(self.num_acoes)
        grad[acao] = 1
        grad -= probs
        self.W_actor[estado, :] += 0.01 * td_error * grad
```

## Capítulo 27: RLHF (Reinforcement Learning from Human Feedback)

```python
class InterfaceRecompensaHumana:
    """Interface para feedback humano"""
    
    def __init__(self):
        self.historico_feedback = []
        self.recompensas_acumuladas = {}
    
    def apresentar_resposta(self, pergunta, resposta):
        """Mostrar resposta e solicitar feedback"""
        print(f"\nPergunta: {pergunta}")
        print(f"Resposta: {resposta}")
        print("\nAvalie: 1=Ruim, 2=Regular, 3=Bom, 4=Excelente")
    
    def receber_feedback(self, rating, sugestao_melhor=None):
        """Receber avaliação do usuário"""
        # Converter rating em recompensa (-1 a 1)
        recompensa = (rating - 2.5) / 1.5
        
        feedback = {
            'rating': rating,
            'recompensa': recompensa,
            'sugestao': sugestao_melhor
        }
        
        self.historico_feedback.append(feedback)
        return recompensa
    
    def extrair_preferencias(self):
        """Analisar padrões de feedback"""
        ratings = [f['rating'] for f in self.historico_feedback]
        media_rating = np.mean(ratings) if ratings else 0
        
        return {
            'media_rating': media_rating,
            'total_feedbacks': len(self.historico_feedback),
            'sugestoes': [f['sugestao'] for f in self.historico_feedback if f['sugestao']]
        }

class ModeloComRLHF:
    """Modelo que aprende com RLHF"""
    
    def __init__(self, tamanho_modelo=1000):
        self.tamanho_modelo = tamanho_modelo
        self.pesos = np.random.randn(tamanho_modelo) * 0.01
        self.interface_feedback = InterfaceRecompensaHumana()
    
    def gerar_resposta(self, pergunta):
        """Gerar resposta (simplificado)"""
        # Simulação: usar pesos para determinar resposta
        score = np.dot(self.pesos[:10], np.random.randn(10))
        respostas = [
            "Resposta conservadora",
            "Resposta média",
            "Resposta criativa"
        ]
        idx = int(np.clip(score, 0, 2))
        return respostas[idx]
    
    def aprender_com_feedback(self, pergunta, resposta, recompensa):
        """Atualizar modelo baseado em recompensa"""
        # Gradiente ascendente de política
        grad = np.random.randn(self.tamanho_modelo) * 0.01
        self.pesos += 0.001 * recompensa * grad
    
    def loop_treinamento_interativo(self, num_iteracoes=5):
        """Loop interativo de aprendizado"""
        print("[=== TREINAMENTO COM FEEDBACK HUMANO ===]")
        
        for i in range(num_iteracoes):
            pergunta = f"Pergunta {i+1}"
            resposta = self.gerar_resposta(pergunta)
            
            self.interface_feedback.apresentar_resposta(pergunta, resposta)
            rating = int(input("Seu rating: "))
            sugestao = input("Sugestão (ou Enter para pular): ") or None
            
            recompensa = self.interface_feedback.receber_feedback(rating, sugestao)
            self.aprender_com_feedback(pergunta, resposta, recompensa)
            
            print(f"[Modelo atualizado com recompensa: {recompensa:.3f}]")
        
        print("\n[Análise de Aprendizado]")
        prefs = self.interface_feedback.extrair_preferencias()
        print(f"Rating médio: {prefs['media_rating']:.2f}")
        print(f"Total de feedbacks: {prefs['total_feedbacks']}")
```

---

# MÓDULO 8: PROJETO FINAL INTEGRADO

## Capítulo 28: Arquitetura do Sistema Completo

```python
class SistemaIAUnificado:
    """
    Integra TODOS os componentes de IA:
    - NLP para entender texto
    - NN para processar
    - Vision para webcam (opcional)
    - RL para aprender com feedback
    - Genéticos para otimizar
    """
    
    def __init__(self, dominio="geral"):
        print("[Inicializando Sistema de IA Integrado...]")
        
        # Componentes NLP
        self.preprocessador = PreprocessadorTexto()
        self.tfidf = TfIdf()
        self.chatbot = ChatbotSimples()
        
        # Componente Neural
        self.rede_neural = RedeNeural([100, 50, 20, 1])
        
        # Componente RL
        self.modelo_rl = ModeloComRLHF()
        
        # Componente Vision (opcional)
        self.detector_faces = None
        self.usar_visao = False
        
        # Histórico
        self.historico_interacoes = []
        self.dominio = dominio
    
    def inicializar_base_conhecimento(self):
        """Configurar base de conhecimento inicial"""
        print("\n[Carregando Base de Conhecimento]")
        
        base = {
            "O que é IA?": "IA é Inteligência Artificial, máquinas que aprendem",
            "Como você funciona?": "Uso NLP para entender, NN para processar, RL para aprender",
            "Qual é seu domínio?": f"Sou especializado em {self.dominio}",
        }
        
        for pergunta, resposta in base.items():
            self.chatbot.adicionar_conhecimento(pergunta, resposta)
        
        print(f"[✓] {len(base)} pares de conhecimento carregados")
    
    def processar_pergunta(self, pergunta):
        """Pipeline completo de processamento"""
        print(f"\n[Processando: {pergunta}]")
        
        # 1. Preprocessamento (NLP)
        tokens = self.preprocessador.processar(pergunta)
        print(f"  → Tokens: {tokens}")
        
        # 2. Buscar resposta
        resposta = self.chatbot.responder(pergunta)
        print(f"  → Resposta: {resposta}")
        
        # Armazenar no histórico
        interacao = {
            'pergunta': pergunta,
            'resposta': resposta,
            'timestamp': np.datetime64('now')
        }
        self.historico_interacoes.append(interacao)
        
        return resposta
    
    def habilitar_visao(self):
        """Ativar processamento de visão"""
        print("\n[Habilitando Visão Computacional]")
        self.detector_faces = DetectorFaces()
        self.usar_visao = True
        print("[✓] Visão ativada - Webcam disponível")
    
    def processar_com_visao(self):
        """Adicionar informação visual às respostas"""
        if not self.usar_visao:
            return None
        
        print("[Capturando imagem...]")
        # Simulação - em produção usar webcam real
        return "Imagem processada com sucesso"
    
    def receber_feedback_do_usuario(self, rating, sugestao=None):
        """Integrar feedback para melhorar modelo"""
        print(f"\n[Feedback recebido: Rating {rating}/5]")
        
        # Atualizar modelo RL
        recompensa = self.modelo_rl.interface_feedback.receber_feedback(rating, sugestao)
        
        # Atualizar modelo neural (simulado)
        if recompensa > 0:
            print("  → Modelo melhorando...")
        else:
            print("  → Modelo ajustando...")
        
        return recompensa
    
    def otimizar_com_geneticos(self):
        """Usar AG para otimizar parâmetros"""
        print("\n[Executando Otimização Genética]")
        
        def funcao_fitness(cromossomo):
            """Avaliar qualidade de configuração"""
            score = np.sum(cromossomo) / len(cromossomo)
            return score
        
        ag = AlgoritmoGenetico()
        ag.criar_populacao_aleatoria(10)
        melhor = ag.evoluir(funcao_fitness, gerações=20)
        
        print(f"[✓] Otimização completa: {melhor}")
        return melhor
    
    def gerar_relatorio(self):
        """Gerar relatório de performance"""
        print("\n" + "="*60)
        print("RELATÓRIO DO SISTEMA DE IA")
        print("="*60)
        
        print(f"Domínio: {self.dominio}")
        print(f"Interações: {len(self.historico_interacoes)}")
        
        if self.historico_interacoes:
            print("\nÚltimas interações:")
            for i, inter in enumerate(self.historico_interacoes[-3:]):
                print(f"  {i+1}. P: {inter['pergunta'][:40]}...")
                print(f"     R: {inter['resposta'][:40]}...")
        
        prefs = self.modelo_rl.interface_feedback.extrair_preferencias()
        print(f"\nFeedback médio: {prefs['media_rating']:.2f}")
        print(f"Total de feedbacks: {prefs['total_feedbacks']}")

# Executar sistema
print("\n" + "="*60)
print("SISTEMA COMPLETO DE IA - DEMONSTRAÇÃO")
print("="*60)

sistema = SistemaIAUnificado(dominio="Python e IA")
sistema.inicializar_base_conhecimento()

# Interações
sistema.processar_pergunta("O que é IA?")
sistema.receber_feedback_do_usuario(5, "Resposta muito boa!")

sistema.processar_pergunta("Como você funciona?")
sistema.receber_feedback_do_usuario(4)

# Otimizar
sistema.otimizar_com_geneticos()

# Relatório
sistema.gerar_relatorio()
```

## Capítulo 29: Integração Completa com Interface

```python
class InterfaceUsuario:
    """Interface interativa completa"""
    
    def __init__(self, sistema_ia):
        self.sistema = sistema_ia
        self.em_execucao = True
    
    def exibir_menu_principal(self):
        """Menu principal"""
        print("\n" + "="*60)
        print("SISTEMA DE IA INTEGRADO - MENU PRINCIPAL")
        print("="*60)
        print("1. Fazer pergunta")
        print("2. Avaliar resposta anterior")
        print("3. Ativar/usar visão computacional")
        print("4. Otimizar modelo (AG)")
        print("5. Ver relatório")
        print("6. Sair")
        print("="*60)
    
    def fazer_pergunta(self):
        """Submenu: pergunta"""
        pergunta = input("\nSua pergunta: ")
        resposta = self.sistema.processar_pergunta(pergunta)
        print(f"\nResposta: {resposta}")
    
    def avaliar_resposta(self):
        """Submenu: feedback"""
        rating = int(input("\nRating (1-5): "))
        sugestao = input("Sugestão de melhoria (opcional): ") or None
        self.sistema.receber_feedback_do_usuario(rating, sugestao)
    
    def usar_visao(self):
        """Submenu: visão"""
        self.sistema.habilitar_visao()
        resultado = self.sistema.processar_com_visao()
        print(f"Resultado: {resultado}")
    
    def executar(self):
        """Loop principal"""
        self.sistema.inicializar_base_conhecimento()
        
        while self.em_execucao:
            self.exibir_menu_principal()
            opcao = input("\nEscolha uma opção: ")
            
            if opcao == "1":
                self.fazer_pergunta()
            elif opcao == "2":
                self.avaliar_resposta()
            elif opcao == "3":
                self.usar_visao()
            elif opcao == "4":
                self.sistema.otimizar_com_geneticos()
            elif opcao == "5":
                self.sistema.gerar_relatorio()
            elif opcao == "6":
                print("Encerrando...")
                self.em_execucao = False
            else:
                print("Opção inválida")
```

## Capítulo 30: Deployment e Otimizações Finais

```python
class SistemaProducao:
    """Preparar sistema para produção"""
    
    def __init__(self, sistema_ia):
        self.sistema = sistema_ia
    
    def salvar_modelo(self, arquivo):
        """Persistir modelo treinado"""
        import pickle
        with open(arquivo, 'wb') as f:
            pickle.dump(self.sistema, f)
        print(f"[✓] Modelo salvo em {arquivo}")
    
    def carregar_modelo(self, arquivo):
        """Carregar modelo salvo"""
        import pickle
        with open(arquivo, 'rb') as f:
            self.sistema = pickle.load(f)
        print(f"[✓] Modelo carregado de {arquivo}")
    
    def criar_api_rest(self):
        """Criar API REST para integração"""
        print("""
        [API REST - Exemplo com Flask]
        
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/api/pergunta', methods=['POST'])
        def processar():
            dados = request.json
            resposta = sistema.processar_pergunta(dados['pergunta'])
            return jsonify({'resposta': resposta})
        
        @app.route('/api/feedback', methods=['POST'])
        def feedback():
            dados = request.json
            sistema.receber_feedback_do_usuario(dados['rating'])
            return jsonify({'status': 'ok'})
        
        if __name__ == '__main__':
            app.run(debug=False, port=5000)
        """)
    
    def benchmarks(self):
        """Testar performance"""
        import time
        
        print("\n[BENCHMARKS]")
        
        # Teste de velocidade
        inicio = time.time()
        for _ in range(100):
            self.sistema.processar_pergunta("teste")
        tempo = time.time() - inicio
        
        print(f"100 perguntas em {tempo:.2f}s ({100/tempo:.0f} req/s)")
        
        # Teste de memória (simplificado)
        num_interacoes = len(self.sistema.historico_interacoes)
        print(f"Interações em memória: {num_interacoes}")

# Exemplo de Deployment
print("\n[DEPLOYMENT]")
sistema_prod = SistemaIAUnificado(dominio="Python")
sistema_prod.inicializar_base_conhecimento()

prod = SistemaProducao(sistema_prod)
prod.salvar_modelo("modelo_ia.pkl")
prod.benchmarks()
prod.criar_api_rest()
```

---

# RESUMO E PRÓXIMOS PASSOS

## Conceitos Cobertos

 **IA Clássica:** Regras, busca, heurísticas  
 **Machine Learning:** Supervisionado, não supervisionado, RL  
 **Redes Neurais:** Perceptron, CNN, RNN, LSTM, Transformers  
 **NLP:** Tokenização, embeddings, TF-IDF, Word2Vec, Chatbots  
 **Genéticos:** Seleção, mutação, crossover, otimização  
 **Visão Computacional:** Processamento, detecção, OCR  
 **Reinforcement Learning:** Q-Learning, Policy Gradient, RLHF  
 **Integração:** Sistema unificado com múltiplos componentes  

## Habilidades Adquiridas

- Implementar algoritmos do zero
- Entender fundamentals matemáticos
- Construir modelos práticos
- Integrar múltiplas técnicas
- Deploy em produção
- Aprender com feedback humano

## Desafios Próximos

1. **Aumentar Dataset:** Treinar com dados reais
2. **Otimizações:** GPU, quantização, pruning
3. **Arquiteturas Avançadas:** Vision Transformers, LLMs
4. **Explicabilidade:** SHAP, LIME, interpretabilidade
5. **Ética em IA:** Bias detection, fairness

## Recursos Adicionais

- **Papers:** arXiv.org, Papers with Code
- **Comunidades:** Reddit r/MachineLearning, Kaggle
- **Cursos:** Fast.ai, Coursera, Udacity
- **Bibliotecas:** TensorFlow, PyTorch, Scikit-learn

**Parabéns! Você agora domina os pilares fundamentais de IA moderna!**
