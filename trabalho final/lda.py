import subprocess
import sys
import os
import requests
import pandas as pd
import numpy as np
import vaex
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Função para instalar pacotes via pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Função para baixar o arquivo com requests
def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Instalar gdown e vaex se não estiverem instalados
try:
    import gdown
    print("gdown já está instalado.")
except ImportError:
    print("gdown não encontrado. Instalando...")
    install('gdown')
    import gdown
    print("gdown instalado com sucesso.")

try:
    import vaex
    print("vaex já está instalado.")
except ImportError:
    print("vaex não encontrado. Instalando...")
    install('vaex')
    import vaex
    print("vaex instalado com sucesso.")

print("Bibliotecas importadas com sucesso.")

file_path = 'dataset_inter_bloco32.csv'

if not os.path.exists(file_path):
    print("Arquivo não encontrado. Baixando o dataset...")
    # URL do arquivo no Google Drive
    file_id = '117hn1Hj9fUl0c51lqCwnDA7GwXd4HnBv'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    download_file(download_url, file_path)
    print("Dataset baixado com sucesso.")

print("Abrindo o dataset...")
dataset_inter = vaex.open(file_path)
print("Dataset carregado.")

print("Selecionando aleatoriamente 50% das linhas do dataset...")
dataset_inter = dataset_inter.sample(frac=0.5)  # selecionando aleatoriamente 50% das linhas do dataset.
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

print("Amostrando 50% dos dados de treino...")
y_train = y_train.set_index(X_train.index)
X_train_sampled = X_train.sample(frac=0.5, random_state=42)
y_train_sampled = y_train.loc[X_train_sampled.index]
print("Amostragem concluída.")

X_train_selected_features = X_train
x_test_selected_features = x_test

# --------------------------------- LDA ----------------------------------
print("Executando LDA...")
#y_train_lda = y_train.ravel()
lda = LinearDiscriminantAnalysis(n_components=len(np.unique(y_train))-1)
X_train_lda = lda.fit_transform(X_train_selected_features, y_train)
X_test_lda = lda.transform(x_test_selected_features)
print("LDA concluído.")


# ------------------------- TREINAMENTO DO MODELO -------------------------
# Medir o tempo de início
start_time = time.time()

print("Treinando o modelo com os melhores parâmetros encontrados...")
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train_lda, y_train)
rf_preds_final = rf.predict(X_test_lda)
print("Treinamento concluído.")

# Medir o tempo de término e calcular o tempo total de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo total de execução: {execution_time:.2f} segundos")

print("Random Forest - RESULTADO FINAL")
print(classification_report(y_test, rf_preds_final))
