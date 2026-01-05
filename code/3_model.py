# Очищаем строки

import re

def clean_address(text):
    text = text.lower()
    # Убираем всё, кроме букв и цифр (опционально)
    text = re.sub(r'[^а-я0-9\s]', '', text)
    return text

# Собственно механизм

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Загружаем эталоны
# refs = ["г. Москва, ул. Ленина, д. 19", "г. Москва, ул. Лесная, д. 5", "г. Казань, ул. Пушкина, д. 1"]
# df_ref = pd.DataFrame(refs, columns=['address'])
df_ref = pd.DataFrame(df_ideal_address['united_addr'])

# 2. Векторизация (разбиваем на кусочки по 3 буквы)
# analyzer='char_wb' создает н-граммы учитывая границы слов
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
ref_vectors = vectorizer.fit_transform(df_ref['united_addr'])

# 3. Индекс для быстрого поиска (Ball Tree или Brute Force для малых данных)
nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1).fit(ref_vectors)

# 4. Функция поиска
def match_address(messy_address):
    # Превращаем вход в вектор
    input_vec = vectorizer.transform([messy_address])
    # Ищем ближайшего соседа
    distances, indices = nbrs.kneighbors(input_vec)

    best_match_index = indices[0][0]
    similarity = 1 - distances[0][0] # переводим расстояние в сходство (0..1)

    return df_ref.iloc[best_match_index]['united_addr'], similarity

# ТЕСТ
bad_address = "ул труд, челяба"
match, score = match_address(bad_address)
print(f"Вход: {bad_address}")
print(f"Найдено: {match} (Сходство: {score:.2f})")
