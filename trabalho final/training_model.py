import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verifica se o pacote gdown está instalado, caso contrário, instala-o
try:
    import gdown
    print("gdown já está instalado.")
except ImportError:
    print("gdown não encontrado. Instalando...")
    install_package('gdown')
    print("gdown instalado com sucesso.")

# Verifica se o pacote vaex está instalado, caso contrário, instala-o
try:
    import vaex
    print("vaex já está instalado.")
except ImportError:
    print("vaex não encontrado. Instalando...")
    install_package('vaex')
    print("vaex instalado com sucesso.")

import pandas as pd
import numpy as np
import os
import vaex
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import time

print("Bibliotecas importadas com sucesso.")

file_path = 'dataset_inter_bloco32.csv'

if not os.path.exists(file_path):
    print("Arquivo não encontrado. Baixando o dataset...")
    gdown.download(id='117hn1Hj9fUl0c51lqCwnDA7GwXd4HnBv', output='dataset_inter.csv', quiet=False)
    print("Dataset baixado com sucesso.")

print("Abrindo o dataset...")
dataset_inter = vaex.open(file_path)
print("Dataset carregado.")

#print("Selecionando aleatoriamente 50% das linhas do dataset...")
#dataset_inter = dataset_inter.sample(frac=0.5)  # selecionando aleatoriamente 50% das linhas do dataset.
print("Trabalhando com 100% das linhas do dataset...")

dataset_inter = dataset_inter.to_pandas_df()
print("Amostragem concluída.")

def replace_values(value):
    if value in [0, 1]:
        return 0
    elif value == 2:
        return 1
    elif value in [3, 4, 5]:
        return 2

print("Substituindo valores na coluna 'MTSChosen'...")
dataset_inter['MTSChosen'] = dataset_inter['MTSChosen'].apply(replace_values)
print("Substituição concluída.")

print("Balanceando as classes...")
minimo = dataset_inter['MTSChosen'].value_counts().min()
dataset_inter = dataset_inter.groupby('MTSChosen').apply(lambda x: x.sample(minimo)).reset_index(drop=True)
print("Balanceamento concluído.")

X_columns = ['TipoPred', 'frame', 'x', 'y', 'Width', 'Height', 'BlockSize', 'Area', 'depth',
             'CUqp', 'splitSeries', 'bdpcmMode', 'SliceIdx', 'MTSAllowed', 'MtsEnabled',
             'ExplicitMtsInter', 'ExplicitMtsIntra', 'ImplicitMTSIntra', 'TSAllowed',
             'LFNST', 'UseISP', 'UseSBT', 'UseMIP', 'IBCFlag', 'mergeFlag', 'mergeIdx',
             'mergeType', 'smvdMode', 'imv', 'bcwIdx', 'SbtIdx', 'SbtPos', 'BitDepth',
             'CurrQP', 'isAffineBlock', 'affineType', 'mv_currentList', 'mv_idx[0]',
             'mv_refIdx[0]', 'mv_type[0]', 'mv_valid[0]', 'mv_hor[0]', 'mv_ver[0]',
             'mv_idx[1]', 'mv_refIdx[1]', 'mv_type[1]', 'mv_valid[1]', 'mv_hor[1]', 'mv_ver[1]',
             'AbsSumResidual', 'AbsSumUltimaLinha', 'AbsSumUltimaColuna', 'lefTopResidual',
             'leftBottomResidual', 'rightTopResidual', 'rightBottomResidual']
y_columns = ['MTSChosen']
X = dataset_inter[X_columns]
y = dataset_inter[y_columns]

print("Dividindo o dataset em treino e teste...")
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print("Divisão concluída.")

if X_train.duplicated().sum() > 0:
    print("Removendo duplicatas do conjunto de treino...")
    X_train = X_train.drop_duplicates()
    indices = X_train.index
    y_train = y_train.loc[indices].reset_index(drop=True)
    print("Duplicatas removidas.")

print("Tratando valores nulos no conjunto de treino e teste...")
X_train = X_train.fillna(999)  # colocando 999 nas linhas nulas
x_test = x_test.fillna(999)
print("Tratamento de valores nulos concluído.")

print("Identificando colunas constantes...")
colunas_constantes = [col for col in X_train.columns if X_train[col].nunique() == 1]
print(f"Colunas constantes: {colunas_constantes}")

print("Removendo colunas constantes...")
X_train = X_train.drop(columns=colunas_constantes)
x_test = x_test.drop(columns=colunas_constantes)
print("Colunas constantes removidas.")

print("Normalizando os dados...")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
x_test_scale = scaler.transform(x_test)
print("Normalização concluída.")

X_train = pd.DataFrame(X_train_scale, columns=X_train.columns)
x_test = pd.DataFrame(x_test_scale, columns=X_train.columns)

print("Selecionando características...")
selected_features = ['frame', 'x', 'y', 'CUqp', 'mv_hor[0]', 'mv_ver[0]', 'mv_hor[1]',
       'mv_ver[1]', 'AbsSumResidual', 'AbsSumUltimaLinha',
       'AbsSumUltimaColuna', 'lefTopResidual', 'leftBottomResidual',
       'rightTopResidual', 'rightBottomResidual']

X_train_selected_features = X_train[selected_features]
x_test_selected_features = x_test[selected_features]
print("Características selecionadas.")

y_train = y_train.set_index(X_train.index)

# ------------------------- TREINAMENTO DO MODELO -------------------------
# Medir o tempo de início
start_time = time.time()

print("Treinando o modelo com os melhores parâmetros encontrados...")
best_params = {'bootstrap': False, 'max_depth': 35, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 278} #Modelo 1
#best_params = {'bootstrap': False, 'max_depth': 41, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 170} #Modelo 2
#best_params = {'bootstrap': False, 'max_depth': 37, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 280} #Modelo 3
rf = RandomForestClassifier(**best_params)
rf.fit(X_train_selected_features, y_train)
rf_preds_final = rf.predict(x_test_selected_features)
print("Treinamento concluído.")

# Medir o tempo de término e calcular o tempo total de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo total de execução: {execution_time:.2f} segundos")

print("Random Forest - RESULTADO FINAL")
print(classification_report(y_test, rf_preds_final))

# Verifica se o pacote matplotlib está instalado, caso contrário, instala-o
try:
    import matplotlib.pyplot as plt
    print("matplotlib já está instalado.")
except ImportError:
    print("matplotlib não encontrado. Instalando...")
    install_package('matplotlib')
    print("matplotlib instalado com sucesso.")

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarizando as classes para a curva ROC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
rf_preds_final_binarized = label_binarize(rf_preds_final, classes=[0, 1, 2])

# Calculando as curvas ROC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], rf_preds_final_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando todas as curvas ROC
plt.figure()
for i in range(y_test_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (área = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # linha diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC para Múltiplas Classes')
plt.legend(loc='lower right')
plt.show()

