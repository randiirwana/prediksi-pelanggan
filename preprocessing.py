import pandas as pd

# Load dataset
df = pd.read_excel("/mnt/data/Online Retail.xlsx")

# Hapus data yang tidak memiliki CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Konversi kolom tanggal ke datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Tambahkan kolom bulan
df['Month'] = df['InvoiceDate'].dt.to_period('M')

# Hitung jumlah pelanggan unik per bulan
monthly_customers = df.groupby('Month')['CustomerID'].nunique().reset_index()
monthly_customers.columns = ['Month', 'CustomerCount']

# Urutkan berdasarkan waktu dan beri nomor bulan
monthly_customers = monthly_customers.sort_values(by='Month')
monthly_customers['MonthNum'] = range(1, len(monthly_customers) + 1)

print(monthly_customers.head())
