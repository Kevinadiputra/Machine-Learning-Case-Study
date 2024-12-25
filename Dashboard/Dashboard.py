import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import warnings
warnings.filterwarnings('ignore')

# Data loading
df = df = pd.read_csv('https://drive.google.com/uc?id=1vSqGTpyfBocrnp4VllFVuBg7QC6B-2jj')

# Header Section
st.title('Case Study Machine Learning GDGoC UNSRI 2024/2025')
st.write('Nama: Kevin Adiputra Mahesa')
st.write('Jurusan: Sistem Komputer')
st.write('Angkatan: 2023')

st.header("Dataset yang Digunakan")
st.write("Dataset yang digunakan dalam case study ini adalah **'Data Science Salaries 2024'**, yang dapat diakses melalui tautan berikut: \
          [Dataset Data Science Salaries 2024 - Kaggle](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)")

st.subheader("Deskripsi Dataset")
st.write("""
    Dataset ini berisi informasi tentang gaji para profesional di bidang data science untuk tahun 2024. Data mencakup berbagai atribut penting seperti:
    - work_year: Tahun kerja.
    - experience_level: Tingkat pengalaman (Entry-Level, Mid-Level, Senior, Executive).
    - employment_type: Jenis pekerjaan (Full-Time, Part-Time, Contract, Freelance).
    - job_title: Pekerjaan.
    - salary: Gaji dalam mata uang lokal.
    - salary_currency: Mata uang gaji.
    - salary_in_usd: Gaji dalam USD.
    - employee_residence: Negara tempat tinggal karyawan.
    - remote_ratio: Persentase kerja jarak jauh (0, 50, atau 100).
    - company_location: Lokasi perusahaan.
    - company_size: Ukuran perusahaan berdasarkan jumlah karyawan (S: Small, M: Medium, L: Large).
""")

# Data Wrangling Section
st.header("1. Data Wrangling")
st.write("### 5 Baris awal Dataset")
st.write(df.head())
st.write("### informasi Dataset")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
st.write(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.")
st.subheader("Pada informasi dasar dataset ini, didapatkan sebagai berikut: ")
st.write("1. Dataset memiliki atribut sebanyak 11 kolom, dan entri sebanyak 14838 baris data.")
st.write("2. Dari informasi dasar tersebut juga dapat diambil kesimpuan bahwa tidak ada nilai yang kosong dari dataset.")
st.write("3. Atribut dataset terdiri dari 4 numerikal, dan 7 kategorikal")

# Data Availability Section
st.header("2. Data Availability")

# Check missing values
st.write("### Mengecek Nilai Kosong")
missing_values = df.isna().sum()
st.write(missing_values[missing_values > 0])
st.write("Terlihat memang benar tidak ada nilai kosong pada Dataset, sehingga tidak diperlukan penanganan lebih lanjut.")

# Check duplicates
st.write("### Mengecek Nilai Duplikat")
duplicates = df.duplicated().sum()
st.write(f"Jumlah duplikat: {duplicates}")
st.write("Terlihat pada data set ini, memiliki duplikat yang sangat banyak. Namun hal itu wajar mengingat Dataset ini memiliki object dan numerikal yang memiliki persentase yang konsisten, seperti jenis perkerjaan, rasio jarak, dan pekerjaan yang sama, namun berbeda individu.")

# Exploration Data Analysis Section
st.header("Exploration Data Analysis")

# Data Description
st.write("### Deskripsi Data")
st.write(df.describe())
st.write("Dari deskripsi data berikut, dapat disimpulkan beberapa hal:")
st.write("1. Rata-rata paa tahun kerja 2023..., yang artinnya sebagian besar tahun kerja berasal dari tahun 2023.")
st.write("2. Rata-Rata Salary in usd adalah 149.874, ")
st.write("3. Kuartil pertama (25%) menunjukkan bahwa 25% dari individu mendapatkan gaji di bawah 102.100, sedangkan kuartil ketiga (75%) menunjukkan bahwa 25% dari individu mendapatkan gaji di atas 185.900. Menjelaskan jika ada kesenjangan yang signifikan pada gaji.")
st.write("4. Pada kuartil 3 (75%) juga pada bagian remote, 100%, menunjukkan bahwa 25% individu yang gajinnya diatas 185.900 bekerja secara remote.")


# Distribution of Numerical Columns
st.write("### Distribusi Kolom Numerikal")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(df["salary_in_usd"], kde=True, ax=axes[0], color="skyblue")
sns.histplot(df["remote_ratio"], kde=True, ax=axes[1], color="lightgreen")
axes[0].set_title("Distribusi Salary in USD")
axes[1].set_title("Distribusi Remote Ratio")
st.pyplot(fig)

# Judul aplikasi
st.title("Insight dari Distribusi Data")
# Insight Distribusi Salary in USD
st.subheader("Distribusi Salary in USD")
st.markdown("- Distribusi yang **skewed ke kanan**, atau **skewed positive**, dapat terlihat ekor yang mengarah ke arah kanan. Hal ini berarti sebagian besar data terkonsentrasi di sebelah kiri.")
st.markdown("- Puncak distribusi berada pada angka sekitar **150.000 USD**, yang artinya banyak data yang berkumpul di sekitar kisaran gaji tersebut.")

# Insight Distribusi Remote Ratio
st.subheader("Distribusi Remote Ratio")
st.markdown("- Distribusi juga **skewed ke kanan**, yang artinya sebagian besar data terkonsentrasi di sebelah kiri.")
st.markdown("- Puncak distribusi berada di nilai **0**, yang artinya mayoritas pekerja melakukan pekerjaannya secara **on site** atau **non-remote**.")

# Correlation Heatmap
st.write("### Heatmap Korelasi Kolom Numerikal")
corr = df[["salary_in_usd", "remote_ratio"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)
st.write("Pada korelasi heatmap ini didapatkan bahwa salary in usd dan remote ratio tidak memiliki korelasi atau hubungan yang kuat.")

# Countplot for Categorical Columns
st.write("### Countplot untuk Kolom Kategorikal")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.countplot(data=df, x="experience_level", ax=axes[0], palette="pastel")
sns.countplot(data=df, x="company_size", ax=axes[1], palette="pastel")
sns.countplot(data=df, x="employment_type", ax=axes[2], palette="pastel")
axes[0].set_title("Pengalaman Kerja")
axes[1].set_title("Ukuran Perusahaan")
axes[2].set_title("Jenis Pekerjaan")
st.pyplot(fig)

# Insight pada Countplot Pengalaman
st.subheader("Countplot Pengalaman")
st.write("""
1. Sebagian besar individu memiliki pengalaman di tingkat **SE (Senior)**, diikuti oleh **MI (Mid)**, **EN (Entry)**, dan **EX (Expert)**.
""")

# Insight pada Countplot Tipe Perusahaan
st.subheader("Countplot Tipe Perusahaan")
st.write("""
2. Sebagian besar individu berasal dari perusahaan **M (Medium)**, diikuti oleh perusahaan **L (Large)**, 
   sedangkan perusahaan **S (Small)** memiliki jumlah individu yang jauh lebih sedikit.
   Hal ini menunjukkan bahwa sebagian besar individu bekerja di perusahaan dengan skala sedang (medium).
""")

# Insight pada Countplot Tipe Pekerjaan
st.subheader("Countplot Tipe Pekerjaan")
st.write("""
3. **Mayoritas atau hampir semua individu bekerja Fulltime (FT)**.
""")

# Salary and Experience Level Boxplot
st.write("### Hubungan Gaji dan Tingkat Pengalaman Kerja")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="experience_level", y="salary_in_usd", palette="pastel")
ax.set_title("Hubungan Gaji dan Tingkat Pengalaman Kerja", fontsize=16)
ax.set_xlabel("Tingkat Pengalaman Kerja", fontsize=12)
ax.set_ylabel("Gaji (USD)", fontsize=12)
st.pyplot(fig)

# Insight pada Boxplot
st.subheader("Berdasarkan Boxplot tersebut, didapatkan sebagai berikut: ")
st.write("""
1. **Semakin tinggi tingkat pengalaman, maka gajinya semakin tinggi.**  
   Hal ini ditunjukkan oleh **median gaji yang lebih tinggi** dibandingkan tingkat pengalaman lainnya.
   
2. **Variasi gaji yang tinggi di setiap tingkat pengalaman.**  
   Dapat dilihat dari rentang antar kuartil (**IQR**) yang cukup lebar, menunjukkan adanya penyebaran gaji yang signifikan.

3. **Outlier pada boxplot.**  
   Menunjukkan adanya individu yang memiliki gaji lebih besar dari tingkat pengalamannya.  
   Hal ini dapat disebabkan oleh faktor-faktor seperti kemampuan khusus atau kebijakan perusahaan tertentu.
""")

# Top 5 Lokasi dengan gaji rata-rata tertinggi
st.write("### 5 Lokasi dengan Gaji Rata-Rata Tertinggi")
average_salary_by_location = df.groupby("company_location")["salary_in_usd"].mean().sort_values(ascending=False)
top_5_locations = average_salary_by_location.head(5)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_5_locations.index, y=top_5_locations.values, palette="viridis")
ax.set_title("5 Lokasi dengan Gaji Rata-Rata Tertinggi", fontsize=16)
ax.set_xlabel("Lokasi Kerja", fontsize=12)
ax.set_ylabel("Gaji Rata-Rata (USD)", fontsize=12)
st.pyplot(fig)

# Insight pada rata-rata gaji
st.subheader("Berdasarkan visualisasi tersebut dapat diambil insight sebagai berikut: ")
st.write("""
1. **Qatar (QA)** memiliki rata-rata gaji tertinggi, yaitu sekitar **300.000 USD**.
   
2. **Israel (IL)** dan **Puerto Rico (PR)** memiliki rata-rata gaji yang lebih rendah daripada **Qatar (QA)**, tetapi masih lebih tinggi daripada **Amerika Serikat (US)** dan **Selandia Baru (NZ)**.
   
3. **Amerika Serikat (US)** dan **Selandia Baru (NZ)** memiliki rata-rata gaji yang lebih rendah dibandingkan dengan **Qatar (QA)**, **Israel (IL)**, dan **Puerto Rico (PR)**.
""")

# Insight Section
st.header("Insight")
st.write("""
    1. Hampir semua individu adalah pekerja Full Time (FT).
    2. Pengalaman memiliki pengaruh terhadap gaji, semakin tinggi pengalaman, semakin tinggi pula gaji. Namun ada beberapa faktor lain yang dapat mempengaruhi seperti lokasi, serta perusahaan.
    3. Kebanyakan individu memiliki pengalaman sebagai Senior.
    4. Rata-rata gaji adalah 149.874 USD.
    5. Mayoritas pekerja bekerja secara on-site dengan rasio remote 0% yang menunjukkan bahwa pekerjaan ini lebih cenderung dilakukan di kantor.
    6. Lokasi dengan gaji rata-rata tertinggi adalah Qatar, diikuti oleh Israel dan Puerto Rico.
""")
