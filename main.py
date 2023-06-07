import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Carregar o dataset do arquivo CSV
df = pd.read_csv('dataset.csv')

# Selecionar as colunas relevantes para prever o salário
columns = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_in_usd','employee_residence']
df = df[columns]
print(df)

# Remover linhas com valores ausentes, se houver
df.dropna(inplace=True)

# Separar os dados de entrada (perfil do profissional) e saída (salário)
X = df[['work_year', 'experience_level', 'employment_type', 'job_title','employee_residence']]
y = df['salary_in_usd']

# Converter strings para números usando LabelEncoder
le_work_year = LabelEncoder()
le_experience_level = LabelEncoder()
le_employment_type = LabelEncoder()
le_job_title = LabelEncoder()
le_employee_residence = LabelEncoder()

X.loc[:, 'work_year'] = X['work_year']
X.loc[:, 'experience_level'] = le_experience_level.fit_transform(X['experience_level'])
X.loc[:, 'employment_type'] = le_employment_type.fit_transform(X['employment_type'])
X.loc[:, 'job_title'] = le_job_title.fit_transform(X['job_title'])
X.loc[:, 'employee_residence'] = le_employee_residence.fit_transform(X['employee_residence'])


# Usar OneHotEncoder para lidar com variáveis categóricas
ohe = OneHotEncoder(handle_unknown='ignore')
X_ohe = ohe.fit_transform(X)

# Normalizar os dados de entrada
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_ohe)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo MLP
modelo = MLPRegressor(hidden_layer_sizes=(10, 10),max_iter=1000, activation="relu", solver="adam", random_state=42)
modelo.fit(X_train, y_train)

# Fazer previsões para novos dados
novo_profissional = [2023, 'MI', 'CT','Data Scientist','US']

novo_profissional_encoded = np.array([
    novo_profissional[0],
    le_experience_level.transform([novo_profissional[1]])[0],
    le_employment_type.transform([novo_profissional[2]])[0],
    le_job_title.transform([novo_profissional[3]])[0],
    le_employee_residence.transform([novo_profissional[3]])[0]
])

novo_profissional_ohe = ohe.transform([novo_profissional_encoded.astype(str)]) 
novo_profissional_scaled = scaler.transform(novo_profissional_ohe)
salario_previsto = modelo.predict(novo_profissional_scaled)
#gerando um valor mais bonitinho 
salario_formatado = [format(valor, ".2f") for valor in salario_previsto]

print("Salário previsto em USD: $", salario_formatado)
