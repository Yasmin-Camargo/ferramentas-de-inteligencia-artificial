import subprocess
import sys
import os

# Função para instalar pacotes via pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Instalar gdown e vaex se não estiverem instalados
try:
    import gdown
except ImportError:
    print("gdown não encontrado. Instalando...")
    install('gdown')
    import gdown
    print("gdown instalado com sucesso.")

try:
    import vaex
except ImportError:
    print("vaex não encontrado. Instalando...")
    install('vaex')
    import vaex
    print("vaex instalado com sucesso.")

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint
import time

print("Bibliotecas importadas com sucesso.")

file_path = 'dataset_inter_bloco32.csv'

# Função para baixar o arquivo com requests
def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

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
dataset_inter = dataset_inter.sample(frac=0.5) # selecionando aleatoriamente 50% das linhas do dataset.
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
X_train = X_train.fillna(999) # colocando 999 nas linhas nulas
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
selected_features = ['frame', 'x', 'y', 'CUqp', 'mv_hor[0]', 'mv_ver[0]', 'mv_hor[1]', 'mv_ver[1]', 'AbsSumResidual', 'AbsSumUltimaLinha', 'AbsSumUltimaColuna', 'lefTopResidual', 'leftBottomResidual', 'rightTopResidual', 'rightBottomResidual']

X_train_selected_features = X_train[selected_features]
x_test_selected_features = x_test[selected_features]
print("Características selecionadas.")

print("Amostrando 50% dos dados de treino...")
y_train = y_train.set_index(X_train.index)
X_train_sampled = X_train_selected_features.sample(frac=0.5, random_state=42)
y_train_sampled = y_train.loc[X_train_sampled.index]
print("Amostragem concluída.")



# Medir o tempo de início
start_time = time.time()

# ------------------------- GRID SEARCH -------------------------

print("Iniciando busca aleatória de hiperparâmetros...")
param_dist = {
    'n_estimators': randint(100, 300),          # Número de árvores na floresta
    'max_features': ['sqrt', 'log2', None],  # Número máximo de características a serem consideradas para dividir um nó
    'max_depth': randint(1, 80),              # Profundidade máxima da árvore
    'min_samples_split': randint(2, 10),        # Número mínimo de amostras necessárias para dividir um nó
    'min_samples_leaf': randint(1, 10),         # Número mínimo de amostras necessárias para estar em um nó folha
    'bootstrap': [True, False]                  # Se as amostras são extraídas com substituição
}

# Criar o modelo RandomForestClassifier
rf = RandomForestClassifier()

# Criar o objeto RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,                        # Número de iterações de busca
    cv=5,                              # Número de folds para validação cruzada
    verbose=2,                          # Verbose level para mais detalhes
    n_jobs=6,                           # Usar todos os núcleos disponíveis
    pre_dispatch=6,                     # Limita o número de jobs despachados
    random_state=42                     # Garantir reprodutibilidade
)

# Ajustar o RandomizedSearchCV aos dados de treino e validação
print("Ajustando o RandomizedSearchCV...")
random_search.fit(X_train_sampled, y_train_sampled)
print("Busca aleatória concluída.")

# Obter os melhores parâmetros e o melhor modelo
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("Melhores parâmetros encontrados:", best_params)

# Medir o tempo de término e calcular o tempo total de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo total de execução GRID SEARCH: {execution_time:.2f} segundos")




# Obter os três melhores modelos e seus parâmetros
results = random_search.cv_results_
top_n = 3  # Número de modelos que você deseja obter

# Ordenar os resultados pela média das pontuações de validação
sorted_indices = np.argsort(results['mean_test_score'])[::-1][:top_n]

print(f"\nTop {top_n} melhores modelos:")

for rank, idx in enumerate(sorted_indices, start=1):
    print(f"\nModelo {rank}:")
    print(f"Score: {results['mean_test_score'][idx]}")
    print(f"Parâmetros: {results['params'][idx]}")


# Avaliar o melhor modelo nos dados de teste
y_pred = best_model.predict(x_test_selected_features)
print(classification_report(y_test, y_pred))





# ------------------------- TREINAMENTO DO MODELO -------------------------
start_time = time.time()

print("Treinando o modelo com os melhores parâmetros encontrados...")
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train_selected_features, y_train)
print("Treinamento concluído.")

# Fazer previsões com o modelo treinado
rf_preds_final = final_model.predict(x_test_selected_features)

end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo total de execução GRID SEARCH: {execution_time:.2f} segundos")

print("Random Forest - RESULTADO FINAL")
print(classification_report(y_test, rf_preds_final))
