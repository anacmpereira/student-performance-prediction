# student-performance-prediction
Previsão da nota de exame de estudantes com Regressão Linear, Random Forest e XGBoost, identificando fatores de desempenho e alunos em risco.
# 🎓 Previsão de Desempenho Estudantil

Projeto de regressão para prever a nota de exame de estudantes e identificar quais fatores a escola e a família podem influenciar, como frequência, horas de estudo e aulas de reforço.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regressão-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)

!(imagens/correlacao.png)
!(imagens/scatterplot.png)

---

## 📌 Objetivo

Prever a nota final no exame (`Exam_Score`) para:

1. **Identificar alunos em risco** de baixo desempenho;
2. **Descobrir quais fatores** podem ser trabalhados por escolas e pais para melhorar os resultados.

## 📊 Base de dados

- **Fonte:** [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyen1/student-performance-factors) 
- **Tamanho:** 6.607 estudantes e 20 atributos
- **Variável alvo:** `Exam_Score` (nota do exame, de 0 a 100)

| Tipo | Variáveis |
|---|---|
| Numéricas | `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity` |
| Ordinais | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`, `Peer_Influence` |
| Nominais | `Extracurricular_Activities`, `Internet_Access`, `Learning_Disabilities`, `School_Type`, `Gender` |

> O arquivo `StudentPerformanceFactors.csv` deve ser baixado do Kaggle e colocado na raiz do projeto.

## 🔧 Etapas do projeto

### 1. Tratamento e limpeza dos dados
- Verificação de tipos, valores únicos e duplicados (nenhum duplicado encontrado)
- **Valores nulos** (< 1,5% em `Teacher_Quality`, `Parental_Education_Level` e `Distance_from_Home`): imputados pela moda, após checar que não se concentravam em nenhum grupo
- **Valor inconsistente** em `Exam_Score` (101, acima do limite da escala): ajustado para 100
- Conversão das categóricas em ordinais (codificação ordenada) e nominais (one-hot encoding)
- **Outliers** analisados por boxplot e método IQR; mantidos por serem valores plausíveis, não erros de registro

### 2. Análise exploratória
- Matriz de correlação de Pearson: `Attendance` (0,58) e `Hours_Studied` (0,45) são as variáveis mais associadas à nota
- Preditoras com baixa correlação entre si (sem indícios de multicolinearidade)
- **PCA** testado, mas descartado: os componentes não tiveram interpretação clara e a variância se distribuiu de forma uniforme, sem ganho de redução de dimensionalidade

### 3. Modelagem
Divisão 80/20 em treino e teste, com padronização ajustada apenas no treino para evitar vazamento de dados.

- Regressão Linear Múltipla (+ Ridge, Lasso e Elastic Net)
- Random Forest (padrão e com `RandomizedSearchCV`)
- XGBoost (padrão e com `RandomizedSearchCV`)
- Validação cruzada de 5 folds

## 🏆 Resultados

**Desempenho no conjunto de teste:**

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| **Regressão Linear Múltipla** | **0,771** | **0,444** | **1,799** |
| Ridge (α = 0,1) | 0,771 | 0,444 | 1,799 |
| Elastic Net (α = 0,1) | 0,765 | 0,530 | — |
| Lasso (α = 0,1) | 0,763 | 0,564 | — |
| XGBoost ajustado | 0,753 | 0,640 | 1,869 |
| Random Forest ajustado | 0,693 | 1,027 | 2,084 |
| Random Forest padrão | 0,667 | 1,086 | 2,169 |
| XGBoost padrão | 0,644 | 1,010 | 2,243 |

**Validação cruzada (5 folds, apenas no treino):**

| Modelo | R² validação | R² treino | MAE | RMSE |
|---|---|---|---|---|
| Regressão Linear | 0,717 ± 0,034 | 0,718 | 0,500 | 2,085 |
| Random Forest ajustado | 0,633 | 0,949 | 1,116 | 2,372 |

### Por que a Regressão Linear Múltipla foi o melhor modelo?

1. **Melhor desempenho preditivo:** maior R² e menores erros no teste, superando XGBoost e Random Forest ajustados.
2. **Sem overfitting:** R² de treino e de validação praticamente iguais (0,718 vs. 0,717), enquanto o Random Forest chegou a 0,949 no treino contra 0,633 na validação.
3. **Coerência com os dados:** a relação entre as variáveis e a nota é predominantemente linear e as preditoras são pouco correlacionadas entre si, então modelos mais complexos não trouxeram ganho.
4. **Interpretabilidade:** com variáveis padronizadas, os coeficientes mostram diretamente o efeito de cada fator na nota.
5. **Simplicidade:** a regularização não melhorou o resultado, então o modelo mais simples é suficiente e mais fácil de explicar e manter.

> **Observação:** o desempenho esperado em dados novos está mais próximo do R² da validação cruzada (~0,72) do que do teste único (0,771).

## 💡 Principais insights

- `Attendance` e `Hours_Studied` são os fatores mais associados à nota, e ambos podem ser influenciados por escola e família.

## ⚠️ Limitações

- Os resultados indicam **associações, não causalidade**.
- O RMSE bem maior que o MAE indica alguns erros isolados, provavelmente ligados a notas atípicas.
- Lasso e Elastic Net foram testados apenas com `alpha = 0,1`, sem ajuste de hiperparâmetros.

## ▶️ Como executar

```bash
# 1. Clone o repositório

# 2. Crie um ambiente virtual

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset do Kaggle e coloque StudentPerformanceFactors.csv na raiz do projeto

# 5. Execute o script
```

## 🛠️ Tecnologias

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost

## 📁 Estrutura do projeto

```
├── students.py                          # Código completo do projeto
├── StudentPerformanceFactors.csv    # Base de dados (baixar do Kaggle)
├── requirements.txt                 # Dependências
└── README.md
```

## 👤 Autor

**Ana Carolina Martins Pereira**
