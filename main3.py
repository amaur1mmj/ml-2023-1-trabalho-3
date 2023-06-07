import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler

# Ler o arquivo CSV
data = pd.read_csv('dataset.csv')

# entrada do usuario
ano = input('Ano de trabalho ')
experience = input('Nivel de cargo : ')
cargo = input('Cargo : ')
local = input('Local : ')
local_emrpesa = input('Local emrpesa : ')
tamnho_empresa = input('Tamanho da emrpesa: ')



# Selecionar as colunas relevantes para treinamento
columns = ['work_year', 'experience_level', 'job_title',
            'salary_in_usd', 'employee_residence', 'company_location', 'company_size']
data = data[columns]

# Lidar com valores ausentes, se necessário
data = data.dropna()

# # Converter variáveis categóricas em numéricas
# categorical_cols = ['experience_level', 'employment_type', 'job_title',
#                      'employee_residence', 'company_location', 'company_size']

categorical_cols = ['experience_level', 'job_title',
                     'employee_residence', 'company_location', 'company_size']

label_encoders = {}
for col in categorical_cols:
    if col != 'work_year':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le


# Aplicar one-hot encoding para as colunas categóricas
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
data_encoded = onehot_encoder.fit_transform(data[categorical_cols])
# print(data_encoded)
column_names = onehot_encoder.get_feature_names_out(categorical_cols)
# print(column_names)
data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
data = pd.concat([data.drop(categorical_cols, axis=1), data_encoded], axis=1)


print(data.head(5).info())

# Dividir os dados em atributos (X) e rótulos (y)
X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']



numeric_cols = ['work_year']  # Adicione outras colunas numéricas, se houver
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


#y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()


# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criar e treinar o modelo MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(120,75,50), max_iter=5000,  
                     tol=0.001, learning_rate_init=0.1, activation='relu',solver='lbfgs',random_state=42, verbose=2)
model.fit(X_train, y_train)

# ,learning_rate='constant'


# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Reverter a escala dos valores previstos para obter os valores originais
#y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calcular o R², MAE e MSE do modelo
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R²:', r2)
print('MAE:', mae)
print('MSE:', mse)


# Exemplo de informações do usuário
user_info1 = {
    'work_year': ano,
    'experience_level': experience,
    #'employment_type': 'FT',
    'job_title': cargo,
    #'salary_currency': 'US',
    'employee_residence': local,
    'company_location': local_emrpesa,
    'company_size': tamnho_empresa
}
user_info = {
    'work_year': 2023,
    'experience_level': 'MI',
    #'employment_type': 'FT',
    'job_title': 'Applied Scientist',
    #'salary_currency': 'US',
    'employee_residence': 'US',
    'company_location': 'ES',
    'company_size': 'L'
}

# Criar DataFrame com as informações do usuário codificadas
encoded_user_data = []
for col in X.columns:
    if col in user_info:
        if col != 'work_year':
            encoded_value = label_encoders[col].transform([user_info[col]])[0]
        else:
            encoded_value = user_info[col]
        encoded_user_data.append(encoded_value)
    else:
        # Se a coluna não estiver presente no conjunto de dados do usuário, preencha com 0
        encoded_user_data.append(0)
user_data = pd.DataFrame([encoded_user_data], columns=X.columns)

# Fazer a previsão do salário
predicted_salary = model.predict(user_data)

# Imprimir a previsão do salário
print('Salário previsto:', predicted_salary)

predicted_salary = model.predict(user_data)
salario_formatado = [format(valor, ".2f") for valor in predicted_salary]
hehe = float(salario_formatado[0])/ 100
print("Salário previsto em USD: $", hehe)
