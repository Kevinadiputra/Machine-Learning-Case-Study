import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Case Study Machine Learning GDGoC UNSRI 2024/2025",
    layout="wide"
)

# Title and Dataset Info
st.title("Case Study Machine Learning GDGoC UNSRI 2024/2025")
st.write("### Nama: Kevin Adiputra Mahesa")
st.write("### Jurusan: Sistem Komputer")
st.write("### Angkatan: 2023")

# Load Dataset
data_url = 'https://drive.google.com/uc?id=1vSqGTpyfBocrnp4VllFVuBg7QC6B-2jj'
st.write("### Dataset: [Data Science Salaries 2024 - Kaggle](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)")
df = pd.read_csv(data_url)

# Show dataset
if st.checkbox("Tampilkan 5 Baris Data Awal"):
    st.dataframe(df.head())

# Dataset Overview
st.write("### Informasi Dataset")
st.write("Dataset memiliki 14838 baris data dan 11 kolom.")
st.write("Tidak terdapat nilai kosong atau atribut yang perlu ditangani lebih lanjut.")

# Distribution of Numerical Columns
st.write("### Distribusi Kolom Numerikal")
numeric_columns = ["salary_in_usd", "remote_ratio"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribusi {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel("Frekuensi", fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# Correlation Heatmap
st.write("### Korelasi Kolom Numerikal")
corr = df[numeric_columns].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
ax.set_title("Correlation Heatmap", fontsize=16)
st.pyplot(fig)

# Countplot for Categorical Columns
st.write("### Distribusi Kolom Kategorikal")
categorical_columns = ["experience_level", "company_size", "employment_type"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(categorical_columns):
    sns.countplot(data=df, x=col, ax=axes[i], palette="pastel")
    axes[i].set_title(f"Count of {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel("Count", fontsize=12)
    axes[i].tick_params(axis="x", rotation=30)
plt.tight_layout()
st.pyplot(fig)

# Boxplot: Salary vs Experience Level
st.write("### Hubungan Gaji dan Tingkat Pengalaman Kerja")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="experience_level", y="salary_in_usd", palette="pastel")
ax.set_title("Hubungan Gaji dan Tingkat Pengalaman Kerja", fontsize=16)
ax.set_xlabel("Tingkat Pengalaman Kerja", fontsize=12)
ax.set_ylabel("Gaji (USD)", fontsize=12)
st.pyplot(fig)

# Top 5 Locations by Average Salary
st.write("### 5 Lokasi dengan Gaji Rata-Rata Tertinggi")
average_salary_by_location = df.groupby("company_location")["salary_in_usd"].mean().sort_values(ascending=False)
top_5_locations = average_salary_by_location.head(5)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_5_locations.index, y=top_5_locations.values, palette="viridis")
ax.set_title("5 Lokasi dengan Gaji Rata-Rata Tertinggi", fontsize=16)
ax.set_xlabel("Lokasi Kerja", fontsize=12)
ax.set_ylabel("Gaji Rata-Rata (USD)", fontsize=12)
st.pyplot(fig)

# Kesimpulan
st.write("### Kesimpulan Insight")
st.markdown("""
1. Hampir semua individu adalah pekerja **FT (Full time)**.
2. Pengalaman memiliki pengaruh terhadap gaji, semakin tinggi pengalaman, semakin tinggi pula gaji.
3. Rata-rata gaji adalah **149.874 USD**.
4. Qatar memiliki rata-rata gaji tertinggi dibandingkan lokasi lainnya.
""")
