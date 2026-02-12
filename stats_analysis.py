# stats_analysis.py

import pandas as pd
import numpy as np

# ====== 1. CSV faylni ochish ======
file_name = "data.csv"

try:
    df = pd.read_csv(file_name)
    print("data.csv muvaffaqiyatli yuklandi ✅")
except FileNotFoundError:
    print("XATO ❌ data.csv topilmadi!")
    exit()

print("\nBirinchi 5 qator:")
print(df.head())


# ====== 2. Statistik ma'lumotlar ======

# describe
describe_df = df.describe()
describe_df.to_csv("describe.csv")
print("describe.csv yaratildi")

# mean
mean_df = df.mean(numeric_only=True).to_frame(name="mean")
mean_df.to_csv("mean.csv")
print("mean.csv yaratildi")

# median
median_df = df.median(numeric_only=True).to_frame(name="median")
median_df.to_csv("median.csv")
print("median.csv yaratildi")

# sum
sum_df = df.sum(numeric_only=True).to_frame(name="sum")
sum_df.to_csv("sum.csv")
print("sum.csv yaratildi")

# min
min_df = df.min(numeric_only=True).to_frame(name="min")
min_df.to_csv("min.csv")
print("min.csv yaratildi")

# max
max_df = df.max(numeric_only=True).to_frame(name="max")
max_df.to_csv("max.csv")
print("max.csv yaratildi")

# std
std_df = df.std(numeric_only=True).to_frame(name="std")
std_df.to_csv("std.csv")
print("std.csv yaratildi")

print("\nBarcha statistik fayllar muvaffaqiyatli saqlandi 🎉")
