# ===== 1. KUTUBXONALAR =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ===== 2. CSV FAYLNI OCHISH =====
# data.csv o'rniga o'z faylingiz nomini yozing
df = pd.read_csv("data.csv")

print("===== DATA HEAD =====")
print(df.head())


# ===== 3. USTUNLAR HAQIDA MA’LUMOT =====
print("\n===== DATA INFO =====")
print(df.info())


# ===== 4. STATISTIK MA’LUMOTLAR =====
print("\n===== DESCRIBE =====")
print(df.describe())


# ===== 5. VIZUALIZATSIYA =====

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Histogramlar")
plt.show()

# Korrelyatsiya xaritasi (faqat sonli ustunlar)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Korrelyatsiya Heatmap")
plt.show()


# ===== 6. NAIVE BAYES MODELI =====

# Oxirgi ustunni target deb olamiz
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Faqat sonli ustunlarni olish (Naive Bayes uchun)
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== NAIVE BAYES NATIJA =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===== 7. 5 TA QONUNIYAT (avtomatik chiqarish) =====

print("\n===== 5 TA QONUNIYAT =====")

desc = df.describe()

print("1. Eng katta o‘rtacha qiymat:",
      desc.loc["mean"].idxmax())

print("2. Eng katta maksimal qiymat:",
      desc.loc["max"].idxmax())

print("3. Eng katta tarqalish (std):",
      desc.loc["std"].idxmax())

print("4. Eng kichik minimal qiymat:",
      desc.loc["min"].idxmin())

print("5. Eng kuchli bog‘lanish (korrelyatsiya):")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)
