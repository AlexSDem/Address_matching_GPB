# df_test с грязными адресами
df_test = pd.DataFrame({'messy_addr': df_ideal_address['keyboard_noise']})

# Для батчевой обработки большого df_test
test_vectors = vectorizer.transform(df_test['messy_addr'])
distances, indices = nbrs.kneighbors(test_vectors)

df_test['matched_addr'] = [df_ref.iloc[i]['united_addr'] for i in indices.flatten()]
df_test['similarity'] = 1 - distances.flatten()
print(df_test)

from rapidfuzz import fuzz, process
import numpy as np

# Получили кандидата от TF-IDF, теперь проверяем точнее
score_levenshtein = fuzz.token_sort_ratio(bad_address, match)
print(f"Levenshtein Score: {score_levenshtein}")

df_test['Leven_score'] = df_test.apply(lambda row: fuzz.token_sort_ratio(row['messy_addr'], row['matched_addr']), axis=1)
# Финальная оценка = среднее TF-IDF + Levenshtein
df_test['final_score'] = (df_test['similarity'] + df_test['Leven_score'] / 100) / 2

df_test[['similarity', 'final_score']] *= 100
df_test
