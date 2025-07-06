import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Meu dataset de exemplo
data = {
    'idade': [22, 45, 25, 35, 52, 23, 40, 60, 48, 30],
    'renda': [3000, 7000, 3200, 4500, 8000, 2900, 5200, 9000, 6200, 4000],
    'libera_credito': [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df[['idade', 'renda']]
y = df['libera_credito']

# Treina modelo
model = RandomForestClassifier(random_state=0)
model.fit(X, y)

# Nova amostra para previsão
nova_amostra = pd.DataFrame({'idade': [38], 'renda': [4800]})

# Previsão
previsao = model.predict(nova_amostra)[0]
if previsao == 1:
    print("Crédito LIBERADO para o cliente.")
else:
    print("Crédito NÃO LIBERADO para o cliente.")

# Explicador SHAP
explainer = shap.Explainer(model, X)
shap_values = explainer(nova_amostra)

# Mostra gráfico SHAP
shap.summary_plot(shap_values.values[:1], nova_amostra, feature_names=X.columns.tolist())

