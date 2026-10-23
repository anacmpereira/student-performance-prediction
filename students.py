# STUDENT PERFORMANCE FACTORS DATASET

# 1. SOBRE A BASE DE DADOS
# Este conjunto de dados reúne informações sobre os fatores que impactam o desempenho acadêmico dos alunos. A base
# contém registros de 6.607 estudantes e inclui 20 atributos relacionados a hábitos de estudo, histórico familiar,
# ambiente escolar, motivação e estilo de vida. A variável alvo é “Exam_Score” (nota do exame), que mede o desempenho
# acadêmico de cada aluno.
#
# Criada para uso em análises de dados educacionais, modelos de aprendizado de máquina e tarefas de previsão de
# desempenho estudantil, a base está disponível de forma gratuita e aberta na plataforma Kaggle. Compreender esses
# fatores pode ajudar professores, pais e pesquisadores a desenvolver melhores estratégias de aprendizagem e sistemas
# de apoio aos estudantes.

# O objetivo desse projeto é prever a nota para identificar alunos em risco e descobrir quais fatores a escola ou os
# pais podem influenciar, como frequência, horas de estudo e reforço.

# --------------------------------------------------
# PARTE 1. -----------------------------------------
# --------------------------------------------------

# 2. IMPORTAÇÃO DE BIBLIOTECAS

# Manipulação de dados e visualização
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento e redução de dimensionalidade
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Separação dos dados, validação cruzada e busca de hiperparâmetros
from sklearn.model_selection import train_test_split, cross_validate, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor

# Métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 3. IMPORTAÇÃO DA BASE DE DADOS

pd.set_option("display.max_columns", None)
base = pd.read_csv("StudentPerformanceFactors.csv", sep = ",")
print(base.head())

# 4. SOBRE AS VARIÁVEIS

# Hours_Studied: Horas de estudo por semana. Numérica
# Attendance: Percentual de aulas frequentadas. Numérica (%)
# Parental_Involvement: Nível de envolvimento dos pais. Categórica
# Access_to_Resources: Disponibilidade de recursos educacionais. Categórica
# Extracurricular_Activities: Participa de atividades extracurriculares. Categórica
# Sleep_Hours: Média de horas de sono por noite. Numérica
# Previous_Scores: Notas em provas anteriores. Numérica
# Motivation_Level: Nível de motivação. Categórica
# Internet_Access: Acesso à internet. Categórica
# Tutoring_Sessions: Número de aulas de reforço por mês. Numérica
# Family_Income: Nível de renda familiar. Categórica
# Teacher_Quality: Qualidade dos professores. Categórica
# School_Type: Tipo de escola. Categórica
# Peer_Influence: Influência dos colegas no desempenho. Categórica
# Physical_Activity: Horas de atividade física por semana. Numérica
# Learning_Disabilities: Possui dificuldades de aprendizagem. Categórica
# Parental_Education_Level: Maior nível de escolaridade dos pais. Categórica
# Distance_from_Home: Distância de casa até a escola. Categórica
# Gender: Gênero. Categórica
# Exam_Score: Nota final no exame. Numérica. VARIÁVEL ALVO.

# 5. TRATAMENTO DE VARIÁVEIS

# 5.1 Valores únicos
print(base.nunique())

# 5.2 Tipo de dados
print(base.dtypes)

# 5.3 Transformação das variáveis
# Ordem das categorias
ordem_lmh = ["Low", "Medium", "High"]

# Variáveis ordinais (Low / Medium / High)
colunas_ordinais = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
]

for col in colunas_ordinais:
    base[col] = pd.Categorical(base[col], categories=ordem_lmh, ordered=True)

# Ordinais com outras categorias
base["Parental_Education_Level"] = pd.Categorical(
    base["Parental_Education_Level"],
    categories=["High School", "College", "Postgraduate"],
    ordered=True,
)
base["Distance_from_Home"] = pd.Categorical(
    base["Distance_from_Home"],
    categories=["Near", "Moderate", "Far"],
    ordered=True,
)
base["Peer_Influence"] = pd.Categorical(
    base["Peer_Influence"],
    categories=["Negative", "Neutral", "Positive"],
    ordered=True,
)

# Variáveis categóricas nominais (sem ordem)
colunas_nominais = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "School_Type",
    "Gender",
]

for col in colunas_nominais:
    base[col] = base[col].astype("category")

# Conferindo os tipos
print(base.dtypes)

# 5.4 Dados duplicados
print("Duplicados: ", base.duplicated().sum())
# Sem dados duplicados

# 5.5 Valores nulos
print("Valores nulos: ", base.isnull().mean() * 100)
# As variáveis 'Teacher_Quality', 'Parental_Education_Level' e 'Distance_from_Home' possuem valores nulos, entretanto
# eles representam menos de 1,5%. Remover essas linhas implicaria perder dados de outras variáveis que estão
# preenchidas. Como os nulos estão espalhados em 3 colunas, o total de linhas removidas chegaria perto de 4%, por isso
# vou imputar. Como as 3 variáveis são categóricas, a melhor opção é imputar pela moda (categoria mais frequente).

colunas_nulos = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]

# Antes de imputar, vou conferir se os valores nulos se concentram em algum grupo
for col in colunas_nulos:
    print(col)
    print(base.groupby(base[col].isnull())["Exam_Score"].mean(), "\n")

# As médias são muito próximas; os valores nulos não se concentram em um grupo específico, então vou imputar pela moda

for col in colunas_nulos:
    moda = base[col].mode()[0]
    base[col] = base[col].fillna(moda)

print(base[colunas_nulos].isnull().sum())

# 6. BALANCEAMENTO DAS VARIÁVEIS CATEGÓRICAS

colunas_categoricas = base.select_dtypes(include="category").columns

for col in colunas_categoricas:
    print(f"--- {col} ---")
    print(base[col].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    print()

# Na análise de balanceamento das variáveis categóricas, observou-se que algumas apresentam distribuição desigual
# entre suas categorias, como Internet_Access, em que a grande maioria dos estudantes possui acesso à internet e
# apenas uma pequena parcela não possui. Entretanto, por se tratar de variáveis preditoras, e não da variável alvo,
# esse desbalanceamento não representa um problema de classe minoritária como ocorreria em tarefas de classificação.
# Além disso, essa distribuição é coerente com a realidade, pois, atualmente, a maior parte dos estudantes tem acesso
# à internet. Dessa forma, optou-se por manter essas variáveis sem qualquer tipo de reamostragem ou agrupamento,
# uma vez que a distribuição observada reflete o comportamento esperado da população estudada.

# 7. VERIFICAÇÃO DAS VARIÁVEIS NUMÉRICAS

print("Describe das variáveis numéricas: ")
print(base.describe())

# A análise das estatísticas descritivas (describe) indica que, em geral, os valores das variáveis estão dentro do
# esperado. A exceção é Exam_Score, cujo valor máximo é 101, o que ultrapassa o limite da escala de avaliação (0 a 100).
# Esse valor provavelmente decorre de um erro de digitação, e por isso será tratado.

print(base[base["Exam_Score"] > 100])

# Vou substituir esse valor por 100 (limite da escala). Como é apenas um registro, isso terá um impacto mínimo

base.loc[base["Exam_Score"] > 100, "Exam_Score"] = 100

# 8. VERIFICAÇÃO DE OUTLIERS - Boxplots e método IQR
#
# # 8.1 'Hours_Studied'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Hours_Studied"])
# plt.title("Boxplot - Hours_Studied")
# plt.ylabel("Horas estudadas")
# plt.show()
#
# # IQR
# Q1 = base["Hours_Studied"].quantile(0.25)
# Q3 = base["Hours_Studied"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Hours_Studied"] < limite_inferior) |
#     (base["Hours_Studied"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # Embora o método IQR tenha identificado 43 valores atípicos em Hours_Studied, eles variam de 1 a 44 horas semanais,
# # intervalo plausível que representa alunos com padrões de estudo distintos. Por não se tratarem de erros de
# # registro e por representarem uma parcela pequena da amostra, optou-se por mantê-los na análise.
#
# # 8.2 'Attendance'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Attendance"])
# plt.title("Boxplot - Attendance")
# plt.ylabel("Attendance")
# plt.show()
#
# # IQR
# Q1 = base["Attendance"].quantile(0.25)
# Q3 = base["Attendance"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Attendance"] < limite_inferior) |
#     (base["Attendance"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # A variável 'Attendance' não possui outliers
#
# # 8.3 'Sleep_Hours'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Sleep_Hours"])
# plt.title("Boxplot - Sleep_Hours")
# plt.ylabel("Sleep_Hours")
# plt.show()
#
# # IQR
# Q1 = base["Sleep_Hours"].quantile(0.25)
# Q3 = base["Sleep_Hours"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Sleep_Hours"] < limite_inferior) |
#     (base["Sleep_Hours"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # A variável 'Sleep_Hours' não possui outliers
#
# # 8.4 'Previous_Scores'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Previous_Scores"])
# plt.title("Boxplot - Previous_Scores")
# plt.ylabel("Previous_Scores")
# plt.show()
#
# # IQR
# Q1 = base["Previous_Scores"].quantile(0.25)
# Q3 = base["Previous_Scores"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Previous_Scores"] < limite_inferior) |
#     (base["Previous_Scores"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # A variável 'Previous_Scores' não possui outliers
#
# # 8.5 'Tutoring_Sessions'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Tutoring_Sessions"])
# plt.title("Boxplot - Tutoring_Sessions")
# plt.ylabel("Tutoring_Sessions")
# plt.show()
#
# # IQR
# Q1 = base["Tutoring_Sessions"].quantile(0.25)
# Q3 = base["Tutoring_Sessions"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Tutoring_Sessions"] < limite_inferior) |
#     (base["Tutoring_Sessions"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # A variável Tutoring_Sessions apresentou 430 valores atípicos pelo método IQR. Ao analisar o boxplot,
# # observa-se que esses valores variam de 0 a 8 sessões de tutoria, intervalo plausível que representa alunos com
# # diferentes padrões de apoio e estudo. Por não se tratarem de erros de registro, optou-se por mantê-los na análise.
#
# # 8.6 'Physical_Activity'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Physical_Activity"])
# plt.title("Boxplot - Physical_Activity")
# plt.ylabel("Physical_Activity")
# plt.show()
#
# # IQR
# Q1 = base["Physical_Activity"].quantile(0.25)
# Q3 = base["Physical_Activity"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Physical_Activity"] < limite_inferior) |
#     (base["Physical_Activity"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)
#
# # A variável 'Physical_Activity' não possui outliers
#
# # 8.7 'Exam_Score'
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot(base["Exam_Score"])
# plt.title("Boxplot - Exam_Score")
# plt.ylabel("Exam_Score")
# plt.show()
#
# # IQR
# Q1 = base["Exam_Score"].quantile(0.25)
# Q3 = base["Exam_Score"].quantile(0.75)
# IQR = Q3 - Q1
# limite_inferior = Q1 - 1.5 * IQR
# limite_superior = Q3 + 1.5 * IQR
# outliers = base[
#     (base["Exam_Score"] < limite_inferior) |
#     (base["Exam_Score"] > limite_superior)
# ]
# print("Número de outliers:", len(outliers))
# print(outliers)

# A variável 'Exam_Score' possui 104 valores atípicos pelo método IQR. Ao analisar o boxplot, observa-se que esses
# valores variam de 0 a 100, intervalo plausível que representa a nota dos alunos. Por não se tratarem de erros de
# digitação, optou-se por mantê-los.

# --------------------------------------------------
# PARTE 2. -----------------------------------------
# --------------------------------------------------

# Matriz de Correlação

plt.figure(figsize=(10, 6))
sns.heatmap(base.corr(numeric_only=True), annot=True)
plt.title("Matriz de correlação")
plt.show()

# A matriz de correlação de Pearson mostrou que as variáveis mais associadas à nota do exame são Attendance (0,58) e
# Hours_Studied (0,45), ambas com relação positiva moderada. Previous_Scores (0,18) e Tutoring_Sessions (0,16)
# apresentaram correlação fraca, enquanto Physical_Activity (0,03) e Sleep_Hours (-0,02) não demonstraram relação
# linear relevante com a variável alvo. As variáveis preditoras apresentaram correlações próximas de zero entre si,
# o que indica ausência de multicolinearidade.

# Relação entre variáveis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.regplot(data=base, x="Attendance", y="Exam_Score", ax=axes[0],
            scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
axes[0].set_title("Attendance x Exam_Score")
sns.regplot(data=base, x="Hours_Studied", y="Exam_Score", ax=axes[1],
            scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
axes[1].set_title("Hours_Studied x Exam_Score")
plt.tight_layout()
plt.show()

# A análise dos gráficos de dispersão (scatterplots), em conjunto com a matriz de correlação de Pearson,
# evidenciou relações positivas entre Attendance e Exam_Score (r = 0,58) e entre Hours_Studied e Exam_Score (r = 0,45),
# ambas classificadas como moderadas. A variável Attendance apresentou a maior correlação com a nota do exame,
# indicando uma associação linear mais forte com o desempenho acadêmico. Visualmente, essa tendência é representada
# pela inclinação ascendente das linhas de regressão em ambos os gráficos, embora a dispersão dos pontos demonstre
# que alunos com valores semelhantes de frequência ou horas de estudo podem apresentar notas distintas. As demais
# variáveis analisadas apresentaram correlações fracas, como Previous_Scores (r = 0,18) e Tutoring_Sessions (r = 0,16),
# ou praticamente nulas, como Physical_Activity (r = 0,03) e Sleep_Hours (r = −0,02). Esses resultados indicam que a
# frequência às aulas e as horas de estudo são as variáveis com maior associação linear individual com a nota do exame
# entre as variáveis avaliadas, embora não expliquem, isoladamente, toda a variabilidade do desempenho acadêmico.
# Ressalta-se que essas associações não implicam causalidade.

# CRIAÇÃO DE MODELOS

# Inicialmente vou começar fazendo um PCA para redução da dimensionalidade

# Preparação
cols_num = ["Hours_Studied", "Attendance", "Sleep_Hours",
            "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

X_pca = StandardScaler().fit_transform(base[cols_num])

pca = PCA()
componentes = pca.fit_transform(X_pca)

# Variância explicada
var_exp = pca.explained_variance_ratio_
var_acum = var_exp.cumsum()

tabela = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(len(var_exp))],
    "Variância (%)": (var_exp * 100).round(1),
    "Acumulada (%)": (var_acum * 100).round(1),
})
print(tabela)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(var_exp) + 1), var_exp * 100, label="Por componente")
ax.plot(range(1, len(var_exp) + 1), var_acum * 100, marker="o", color="black", label="Acumulada")
ax.axhline(80, linestyle="--", color="gray")
ax.text(len(var_exp), 81, "80%", ha="right", color="gray")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Variância explicada (%)")
ax.set_title("Variância explicada por componente")
ax.legend()
plt.tight_layout()
plt.show()

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=cols_num,
    columns=[f"PC{i+1}" for i in range(len(cols_num))],
)
print(loadings.round(2))

plt.figure(figsize=(9, 5))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Contribuição das variáveis em cada componente (loadings)")
plt.tight_layout()
plt.show()

# A análise das cargas (loadings) mostrou que cada componente principal combina diversas variáveis com pesos semelhantes,
# sem um padrão interpretável, o que é esperado diante da baixa correlação entre as preditoras. A única exceção parcial
# foi o PC4, com maior contribuição de Hours_Studied (0,78). Como os componentes não possuem interpretação clara
# e a variância se distribui de forma praticamente uniforme entre eles, o PCA não oferece vantagem em redução de
# dimensionalidade nem em interpretabilidade, e foi mantida a modelagem com as variáveis originais.

# --------------------------------------------------
# MODELO 1. REGRESSÃO LINEAR -----------------------
# --------------------------------------------------

# REGRESSÃO LINEAR
# Separação treino/teste e pré-processamento

# Passo 1: Pré-processamento
# Variáveis
cols_num = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

cols_ord = ["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality",
            "Parental_Education_Level", "Distance_from_Home", "Peer_Influence"]

cols_nom = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities", "School_Type", "Gender"]

base_modelo = base.copy()

for col in cols_ord:
    base_modelo[col] = base_modelo[col].cat.codes

base_modelo = pd.get_dummies(base_modelo, columns=cols_nom, drop_first=True, dtype=int)
print("Base modelo:")
print(base_modelo)

# Passo 2: Separação treino/teste e padronização
# Separar as variáveis independentes da variável dependente
X = base_modelo.drop(columns="Exam_Score")  # Variáveis independentes
y = base_modelo["Exam_Score"]  # Variável dependente (rótulo)

# Separando os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fazendo a padronização agora para evitar vazamento
# O scaler aprende (fit) apenas com os dados de treino e depois só transforma (transform) o teste
scaler = StandardScaler()
X_train_padronizado = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_padronizado = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# Conferindo os tamanhos (treino: 5285 linhas | teste: 1322 linhas)
print("Treino:", X_train_padronizado.shape, Y_train.shape)
print("Teste:", X_test_padronizado.shape, Y_test.shape)

# Passo 3: Regressão linear múltipla
regressao_multipla = LinearRegression()
regressao_multipla.fit(X_train_padronizado, Y_train)

print("Intercepto da regressão: ", regressao_multipla.intercept_)

# Como as variáveis foram padronizadas, cada coeficiente representa o efeito na nota de +1 desvio-padrão da variável,
# o que permite comparar a importância das variáveis entre si
coeficientes = pd.Series(regressao_multipla.coef_, index=X.columns)
print("Coeficientes da regressão múltipla:")
print(coeficientes.sort_values(key=abs, ascending=False).round(3))

# Passo 4: Avaliação do modelo
y_pred_train = regressao_multipla.predict(X_train_padronizado)
y_pred_test = regressao_multipla.predict(X_test_padronizado)

print("R2 treino: ", r2_score(Y_train, y_pred_train))
print("R2 teste: ", r2_score(Y_test, y_pred_test))
print("MAE teste: ", mean_absolute_error(Y_test, y_pred_test))
print("RMSE teste: ", np.sqrt(mean_squared_error(Y_test, y_pred_test)))

# O modelo de regressão linear múltipla apresentou bom desempenho (R² teste = 0,77), sem sinais de overfitting.
# O erro absoluto médio de 0,44 ponto indica boa precisão na maioria dos casos, enquanto o RMSE de 1,80 sugere a
# presença de alguns erros isolados de maior magnitude, possivelmente associados a outliers.

# Melhorando o modelo
# Passo 5: Regressão Ridge, Lasso e Elastic Net

# RIDGE
# Criando e treinando o modelo Ridge
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_padronizado, Y_train)

# Fazendo previsões no conjunto de teste
y_pred_ridge = ridge_model.predict(X_test_padronizado)

# Avaliando o modelo
print("REGRESSÃO RIDGE - RESULTADOS NO TESTE")
print(f"R²: {r2_score(Y_test, y_pred_ridge):.3f}")
print(f"MAE: {mean_absolute_error(Y_test, y_pred_ridge):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, y_pred_ridge)):.3f}")

# A regressão Ridge, ajustada com parâmetro de regularização alpha = 0,1, apresentou R² de 0,771, MAE de
# 0,444 e RMSE de 1,799 no conjunto de teste. Esses resultados foram praticamente idênticos aos obtidos pela regressão
# linear múltipla original, que apresentou R² de 0,771, MAE de 0,444 e RMSE de 1,799. A diferença mínima no RMSE indica
# que a aplicação da regularização não proporcionou uma melhoria relevante no desempenho preditivo do modelo.

# LASSO
# Criando e treinando o modelo Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_padronizado, Y_train)

y_pred_lasso = lasso_model.predict(X_test_padronizado)

# Avaliando o modelo
print("REGRESSÃO LASSO - RESULTADOS NO TESTE")
print(f"R²: {r2_score(Y_test, y_pred_lasso):.3f}")
print(f"MAE: {mean_absolute_error(Y_test, y_pred_lasso):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, y_pred_lasso)):.3f}")

# ELASTIC NET
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_model.fit(X_train_padronizado, Y_train)

# Fazendo previsões
y_pred_elastic_net = elastic_net_model.predict(X_test_padronizado)

# Avaliando o modelo
print("REGRESSÃO ELASTIC NET - RESULTADOS NO TESTE")
print(f"R²: {r2_score(Y_test, y_pred_elastic_net):.3f}")
print(f"MAE: {mean_absolute_error(Y_test, y_pred_elastic_net):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, y_pred_elastic_net)):.3f}")

# Os modelos regularizados não superaram a regressão linear múltipla. O Ridge apresentou desempenho idêntico (R² = 0,771),
# enquanto Lasso (R² = 0,763) e Elastic Net (R² = 0,765) tiveram MAE maiores (0,564 e 0,530, contra 0,444). Como o modelo
# linear não apresenta overfitting e o número de variáveis é pequeno em relação ao de observações, a regularização não traz
# ganho, e a penalização L1 tende a remover sinal útil, aumentando o erro típico das previsões.

# Passo 6: Validação cruzada

# Pipeline
pipeline = Pipeline([("scaler", StandardScaler()), ("modelo", LinearRegression()),])

# Definindo a divisão dos dados (folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Rodando a validação cruzada
resultados = cross_validate(
    pipeline, X_train, Y_train, cv=kf,
    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    return_train_score=True,
)

# Resultados
print("R2 de cada fold (validação):", resultados["test_r2"].round(3))
print("R2 médio: %.3f +/- %.3f" % (resultados["test_r2"].mean(), resultados["test_r2"].std()))
print("R2 treino médio: %.3f" % resultados["train_r2"].mean())
print("MAE médio: %.3f" % -resultados["test_neg_mean_absolute_error"].mean())
print("RMSE médio: %.3f" % -resultados["test_neg_root_mean_squared_error"].mean())

# A validação cruzada com 5 folds resultou em R² médio de 0,717 (±0,034), praticamente igual ao R² de treino (0,718),
# indicando ausência de overfitting. O MAE médio de 0,50 contra um RMSE de 2,09 sugere que o erro está concentrado em
# poucas observações atípicas.

# O modelo inicial apresentou R² de 0,717 no treino e 0,771 no conjunto de teste. A validação cruzada com 5 folds
# resultou em R² médio de 0,717 (±0,034), indicando que o resultado do teste único foi ligeiramente favorável e que o
# desempenho esperado do modelo em dados novos é de aproximadamente 0,72. Em ambas as avaliações, o MAE é muito menor
# que o RMSE, o que sugere que o erro se concentra em poucas observações atípicas.

# --------------------------------------------------
# MODELO 2. RANDOM FOREST --------------------------
# --------------------------------------------------

# RANDOM FOREST

# Passo 1: Random Forest com hiperparâmetros padrão

# O Random Forest geralmente se adapta melhor aos dados não padronizados. Como ele é baseado em árvores de decisão, é
# menos sensível à escala dos dados

# Iniciando o modelo Random Forest
rf_model_reg = RandomForestRegressor(random_state=42)

# Treinando o modelo
rf_model_reg.fit(X_train, Y_train)

# Fazendo previsões
Y_pred_reg = rf_model_reg.predict(X_test)

# Trazendo as métricas para avaliar o modelo
mse = mean_squared_error(Y_test, Y_pred_reg)
mae = mean_absolute_error(Y_test, Y_pred_reg)
r2 = r2_score(Y_test, Y_pred_reg)

print("Métricas para o modelo RandomForest: ")
print(f"RMSE: {mse ** 0.5}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

# O Random Forest apresentou desempenho inferior à regressão linear múltipla (R² de 0,667 contra 0,771; MAE de 1,086
# contra 0,444), indicando que a relação entre as variáveis e a nota é predominantemente linear, o que favorece o modelo
# linear, além de ser mais interpretável.

# Passo 2: Melhorando os hiperparâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model_reg_best = RandomForestRegressor(random_state=42)

# Definindo o RandomizedSearchCV para regressão:
random_search_reg = RandomizedSearchCV(estimator=rf_model_reg_best, param_distributions=param_grid,
                                       n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='neg_mean_squared_error')

random_search_reg.fit(X_train, Y_train)

# Printando os melhores hiperparâmetros
best_params_reg = random_search_reg.best_params_
print(f"Melhores Hiperparâmetros: {best_params_reg}")

# Treinando o modelo com os melhores hiperparâmetros
best_rf_model_rg = random_search_reg.best_estimator_
best_rf_model_rg.fit(X_train, Y_train)

# Fazendo as previsões
Y_pred_reg2 = best_rf_model_rg.predict(X_test)

# Trazendo as métricas
mse = mean_squared_error(Y_test, Y_pred_reg2)
mae = mean_absolute_error(Y_test, Y_pred_reg2)
r2 = r2_score(Y_test, Y_pred_reg2)

print("Métricas para o modelo RandomForest com melhora de HIPERPARÂMETROS: ")
print(f"Melhores hiperparâmetros: {random_search_reg.best_params_}")
print(f"RMSE: {mse ** 0.5}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

# O ajuste de hiperparâmetros com RandomizedSearchCV melhorou o Random Forest (R² de 0,667 para 0,693; RMSE de
# 2,169 para 2,084), mas o modelo permaneceu inferior à regressão linear múltipla (R² de 0,771). Isso indica que a
# relação entre as variáveis e a nota é predominantemente linear.

# Passo 3: VALIDAÇÃO CRUZADA + MELHORA DE HIPERPARÂMETROS

scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]

# Random Forest com os melhores hiperparâmetros encontrados na busca
rf_ajustado = RandomForestRegressor(random_state=42, **random_search_reg.best_params_)

# Random Forest com hiperparâmetros padrão (para comparar o efeito do ajuste)
rf_padrao = RandomForestRegressor(random_state=42)

resultados_rf_ajustado = cross_validate(rf_ajustado, X_train, Y_train, cv=kf, scoring=scoring,
                                        return_train_score=True, n_jobs=-1)
resultados_rf_padrao = cross_validate(rf_padrao, X_train, Y_train, cv=kf, scoring=scoring,
                                      return_train_score=True, n_jobs=-1)

# Resultados do Random Forest ajustado, no mesmo formato da regressão linear
print("RANDOM FOREST AJUSTADO")
print("R2 de cada fold (validação):", resultados_rf_ajustado["test_r2"].round(3))
print("R2 médio: %.3f +/- %.3f" % (resultados_rf_ajustado["test_r2"].mean(), resultados_rf_ajustado["test_r2"].std()))
print("R2 treino médio: %.3f" % resultados_rf_ajustado["train_r2"].mean())
print("MAE médio: %.3f" % -resultados_rf_ajustado["test_neg_mean_absolute_error"].mean())
print("RMSE médio: %.3f" % -resultados_rf_ajustado["test_neg_root_mean_squared_error"].mean())

# Tabela comparativa dos três modelos (resultados = validação cruzada da regressão linear, feita antes)
def resumo(nome, res):
    return {
        "Modelo": nome,
        "R2 validação": res["test_r2"].mean(),
        "R2 desvio": res["test_r2"].std(),
        "R2 treino": res["train_r2"].mean(),
        "MAE": -res["test_neg_mean_absolute_error"].mean(),
        "RMSE": -res["test_neg_root_mean_squared_error"].mean(),
    }

tabela_cv = pd.DataFrame([
    resumo("Regressão linear", resultados),
    resumo("Random Forest padrão", resultados_rf_padrao),
    resumo("Random Forest ajustado", resultados_rf_ajustado),
]).round(3)

print("\nCOMPARAÇÃO (validação cruzada, 5 folds, só no treino)")
print(tabela_cv.to_string(index=False))

# O ajuste dos hiperparâmetros melhorou modestamente o desempenho do modelo Random Forest, elevando o R² médio na
# validação cruzada de 0,618 para 0,633 e reduzindo o MAE de 1,174 para 1,116 e o RMSE de 2,422 para 2,372. Entretanto,
# o R² médio de treino permaneceu elevado (0,949), em comparação ao R² médio de validação (0,633), indicando sinais de
# overfitting, ou seja, dificuldade de generalização para dados não utilizados no treinamento. Em comparação com a
# regressão linear múltipla, que apresentou R² médio de validação de 0,717, MAE de 0,500 e RMSE de 2,085, o Random
# Forest ajustado apresentou desempenho inferior em todas as métricas avaliadas. Assim, até o momento, a regressão linear
# múltipla é o modelo mais adequado entre os testados para a previsão das notas dos estudantes, apresentando melhor
# desempenho preditivo e maior consistência entre os resultados de treino e validação.

# --------------------------------------------------
# MODELO 3. XGBOOST --------------------------------
# --------------------------------------------------

# XGBOOST

# Passo 1: XGBoost com hiperparâmetros padrão
# Criação do modelo
model_xgboost = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse'
)

print(model_xgboost.objective)

# Treinando o modelo
model_xgboost.fit(X_train, Y_train)

# Fazendo as previsões
Y_pred = model_xgboost.predict(X_test)

# Avaliação do modelo
rmse_xgboost = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae_xgboost = mean_absolute_error(Y_test, Y_pred)
r2_xgboost = r2_score(Y_test, Y_pred)

print("Métricas para o modelo XGBoost:")
print(f"RMSE: {rmse_xgboost:.3f}")
print(f"MAE: {mae_xgboost:.3f}")
print(f"R²: {r2_xgboost:.3f}")

# O modelo XGBoost apresentou R² de 0,644, MAE de 1,010 e RMSE de 2,243 no conjunto de teste, indicando capacidade de
# explicar parte da variação das notas dos estudantes. Em comparação com o Random Forest ajustado (R² de 0,693, MAE de
# 1,027 e RMSE de 2,084), o XGBoost apresentou R² menor e RMSE maior, com MAE ligeiramente menor. A regressão linear
# múltipla permaneceu como o modelo de melhor desempenho entre os avaliados, apresentando R² médio de validação cruzada
# de 0,717, MAE de 0,500 e RMSE de 2,085. Assim, os resultados obtidos até o momento favorecem a regressão linear
# múltipla para a previsão das notas dos estudantes, sendo necessária uma avaliação sob o mesmo procedimento de
# validação para confirmar essa comparação.

# Melhorando o modelo
# Passo 2: VALIDAÇÃO CRUZADA + MELHORA DE HIPERPARÂMETROS
# Definindo os hiperparâmetros que serão testados
param_grid_xgb = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10]
}

# Criando o modelo XGBoost
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)

# Definindo a validação cruzada de 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configurando a busca de hiperparâmetros
random_search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid_xgb,
    n_iter=30,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# Executando a busca somente nos dados de treino
random_search_xgb.fit(X_train, Y_train)

# Exibindo os melhores hiperparâmetros
print("Melhores hiperparâmetros do XGBoost:")
print(random_search_xgb.best_params_)

# Exibindo o RMSE médio da validação cruzada
rmse_cv_xgb = np.sqrt(-random_search_xgb.best_score_)

print(f"RMSE médio da validação cruzada: {rmse_cv_xgb:.3f}")

# AVALIAÇÃO FINAL DO XGBOOST AJUSTADO
# Recuperando o melhor modelo encontrado
best_xgb_model = random_search_xgb.best_estimator_

# Fazendo previsões no conjunto de teste
Y_pred_xgb_ajustado = best_xgb_model.predict(X_test)

# Calculando as métricas
rmse_xgb = np.sqrt(
    mean_squared_error(Y_test, Y_pred_xgb_ajustado)
)

mae_xgb = mean_absolute_error(
    Y_test, Y_pred_xgb_ajustado
)

r2_xgb = r2_score(
    Y_test, Y_pred_xgb_ajustado
)

# Exibindo os resultados
print("\nXGBOOST AJUSTADO - RESULTADOS NO TESTE")
print(f"RMSE: {rmse_xgb:.3f}")
print(f"MAE: {mae_xgb:.3f}")
print(f"R²: {r2_xgb:.3f}")

# Em relação ao XGBoost inicial, o modelo ajustado melhorou nas três métricas: aumentou o R² de 0,644 para 0,753,
# reduziu o MAE de 1,010 para 0,640 e reduziu o RMSE de 2,243 para 1,869.

# Organizando os resultados da busca
resultados_xgb = pd.DataFrame(random_search_xgb.cv_results_)

# Mostrando os cinco melhores resultados
print(
    resultados_xgb.sort_values("rank_test_score")[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].head()
)

# Após o ajuste dos hiperparâmetros por meio do RandomizedSearchCV, o modelo XGBoost apresentou RMSE médio de 2,150 na
# validação cruzada. No conjunto de teste, obteve R² de 0,753, MAE de 0,640 e RMSE de 1,869, indicando melhora em relação
# à versão inicial do modelo. Entretanto, a Regressão Linear apresentou desempenho ligeiramente superior no conjunto
# de teste, com R² de 0,771, MAE de 0,444 e RMSE de 1,799. Esses resultados indicam que a Regressão Linear explica
# uma proporção um pouco maior da variabilidade das notas e apresenta menores erros médios de previsão. Assim, apesar
# da melhoria obtida com o ajuste dos hiperparâmetros do XGBoost, a maior complexidade desse algoritmo não proporcionou
# ganhos de desempenho para este conjunto de dados, tornando a Regressão Linear o modelo mais adequado entre os avaliados,
# considerando as métricas do conjunto de teste.

# GRÁFICOS FINAIS:
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real vs previsto
axes[0].scatter(Y_test, y_pred_test, alpha=0.4)
axes[0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
axes[0].set_xlabel("Nota real")
axes[0].set_ylabel("Nota prevista")
axes[0].set_title("Regressão linear: real vs previsto (teste)")

# Coeficientes padronizados (top 10)
top = coeficientes.sort_values(key=abs, ascending=False).head(10)[::-1]
axes[1].barh(top.index, top.values)
axes[1].set_title("Top 10 coeficientes padronizados")
axes[1].set_xlabel("Efeito na nota (+1 desvio-padrão)")

plt.tight_layout()
plt.show()

# Gráfico 1 — Valores reais vs. valores previstos: observa-se que a maioria dos pontos está próxima à linha tracejada
# vermelha, que representa a correspondência ideal entre as notas reais e previstas. Isso indica que o modelo apresenta
# boas previsões para grande parte das observações, especialmente para notas reais aproximadamente entre 55 e 78 pontos.
# Entretanto, alguns pontos se afastam consideravelmente dessa linha, sobretudo para notas reais superiores a 80, nas
# quais o modelo prevê valores significativamente menores que os observados. Esse padrão sugere que o modelo apresenta
# dificuldades para estimar notas mais elevadas, tendendo a subestimar esses valores. Portanto, apesar do bom
# alinhamento de grande parte dos pontos, existem erros de previsão importantes em algumas observações, que devem
# ser considerados na avaliação do desempenho do modelo.
#
# Gráfico 2 — Top 10 coeficientes padronizados: o gráfico apresenta a magnitude dos coeficientes padronizados das dez
# variáveis com maior contribuição estimada para a previsão da nota. Attendance apresenta o maior coeficiente,
# de aproximadamente 2,3 desvios-padrão na nota para cada aumento de um desvio-padrão na variável preditora, mantendo
# as demais variáveis constantes. Em seguida, destaca-se Hours_Studied, com coeficiente próximo de 1,75. Essas duas
# variáveis apresentam efeitos estimados consideravelmente superiores aos das demais, reforçando sua relevância para o
# modelo. Access_to_Resources, Previous_Scores e Parental_Involvement aparecem em posições intermediárias, enquanto
# Tutoring_Sessions, Family_Income, Peer_Influence, Parental_Education_Level e Motivation_Level apresentam coeficientes
# de menor magnitude entre as variáveis exibidas. Como o gráfico mostra apenas a magnitude dos coeficientes, sem indicar
# seus sinais, não é possível determinar, a partir dele isoladamente, se cada associação é positiva ou negativa.

# --------------------------------------------------
# CONCLUSÃO ----------------------------------------
# --------------------------------------------------

# O objetivo do projeto foi prever a nota do exame (Exam_Score) para identificar alunos em risco e descobrir quais
# fatores a escola e os pais podem influenciar. Foram avaliados três modelos: Regressão Linear Múltipla (com as
# variações Ridge, Lasso e Elastic Net), Random Forest e XGBoost, estes dois últimos com ajuste de hiperparâmetros.

# O melhor modelo foi a Regressão Linear Múltipla, pelos seguintes motivos:

# 1. Melhor desempenho preditivo: no conjunto de teste, obteve o maior R² (0,771) e os menores erros (MAE de 0,444 e
#    RMSE de 1,799), superando o XGBoost ajustado (R² 0,753; MAE 0,640; RMSE 1,869) e o Random Forest ajustado
#    (R² 0,693; MAE 1,027; RMSE 2,084).

# 2. Consistência e ausência de overfitting: na validação cruzada com 5 folds, o R² médio de validação (0,717)
#    ficou praticamente igual ao de treino (0,718). Já o Random Forest apresentou R² de treino de 0,949 contra 0,633
#    na validação, indicando overfitting claro.

# 3. Coerência com a estrutura dos dados: a matriz de correlação já indicava relações lineares entre a nota e
#    variáveis como Attendance e Hours_Studied, e as preditoras apresentaram baixa correlação entre si, sem
#    multicolinearidade. Nesse cenário, modelos mais complexos, capazes de capturar relações não lineares e
#    interações, não trouxeram ganho.

# 4. Interpretabilidade: como as variáveis foram padronizadas, os coeficientes mostram diretamente o efeito de cada
#    fator na nota, o que responde ao objetivo do projeto de identificar o que escola e pais podem influenciar.
#   Algo que Random Forest e XGBoost não oferecem de forma tão direta.
#
# 5. Simplicidade: a regularização (Ridge, Lasso, Elastic Net) não trouxe melhora, então o modelo mais simples é
#    suficiente, com menor custo computacional e mais fácil de explicar e manter.

# Principais fatores: Attendance e Hours_Studied foram os que mais se associaram à nota, seguidos por Previous_Scores
# e Tutoring_Sessions. Como frequência, horas de estudo e reforço são fatores que escola e família podem influenciar,
# são os pontos de atuação mais promissores para apoiar alunos em risco.

# Limitações: o erro típico (MAE) é baixo, mas o RMSE maior indica alguns erros isolados, provavelmente ligados a
# notas atípicas. Além disso, o desempenho esperado em dados novos é mais próximo do R² da validação cruzada (~0,72)
# do que do teste único (0,771). Como a base é de uso educacional/aberto, os resultados mostram associações e não
# devem ser interpretados como causalidade. Como próximos passos, seria possível ajustar os hiperparâmetros do Lasso
# e do Elastic Net e testar interações entre variáveis.