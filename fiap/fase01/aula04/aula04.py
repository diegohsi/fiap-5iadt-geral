from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Passo 1: Dados iniciais (textos e categorias)
textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]

# Passo 2: Vetorização dos textos
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)  # Transforma textos em matriz numérica

# Passo 3: Divisão dos dados (50% treino, 50% teste)
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)

# Passo 4: Treinamento do modelo Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)  # Treina o modelo com os dados do treino

# Passo 5: Predição e avaliação
y_pred = clf.predict(X_test)  # Previsão das categorias no conjunto de teste
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")  # Calcula e exibe a acurácia
