import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/GlobalLandTemperaturesByCountry.csv")

# cleaning the data
df['dt'] = pd.to_datetime(df['dt'])
df = df.dropna(subset=['AverageTemperature'])
df = df[df['dt'].dt.year >= 1900]
df = df.dropna(subset=['Country'])
df = df.drop_duplicates(subset = ['dt','Country'], keep = 'first')

# now to plot the data

# to map a histogram of the distribution of average land temperatures

plt.hist(df['AverageTemperature'], bins=40, color='#1f77b4',  alpha=0.9)
plt.title("Distribution of Average Land Temperatures (1900–2013)", fontsize=16, fontweight='bold')
plt.xlabel("Average Temperature (°C)", fontsize=12)
plt.ylabel("Number of Records", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('#555')

plt.tight_layout()
plt.show()



#to map a line chart of top 10 hottest countries
df['dt'] = pd.to_datetime(df['dt'])
df['Year'] = df['dt'].dt.year
top_10 = df.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False).head(10).index.tolist()

if 'India' not in top_10:
    top_10.append('India')

df_selected = df[df['Country'].isin(top_10)]
grouped = df_selected.groupby(['Year', 'Country'])['AverageTemperature'].mean().unstack()

plt.figure(figsize=(14, 6))
for country in grouped.columns:
    plt.plot(grouped.index, grouped[country], label=country)

plt.title("India vs Top 10 Hottest Countries (Avg Temperature Over Years)")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.show()





