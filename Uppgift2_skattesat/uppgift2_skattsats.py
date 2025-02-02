#Uppgift 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import alpha

# Ställ in Pandas för att visa fler rader
pd.options.display.max_rows = 1500

# Läs in CSV-filen
df = pd.read_csv("skattsats2025.csv", encoding="utf-8-sig", sep=";", decimal=".", skipinitialspace=True, low_memory=False)
# df är en variabel som innehåller data från dataportal.se
#encoding='latin1':Många CSV-filer från europeiska källor är kodade med latin1(alt. encoding='cp1252', "utf-8-sig")
# header=0 refrerar att första raden i datafilen är en rubrikrad och 0 refererar att första raden börjar med 0
# skipinitialspace för att ta bort mellanslag i kolumnnamn
# low_memory för att inte ta bort minne för att läsa datafiler
# decimal= "." för att konvertera datatyper till decimal_tal (, till . för decimal_tal)

# Utforska och visa dataframe:
print(f"\nFörsta 5 rader i df:", df.head(5))
print(f"\nSista 5 rader i df:", df.tail(5))
print(f"\nAntal kolumner: {df.shape[1]}")
print(f"\nAntal rader: {df.shape[0]}")
print(f"\nKolumnnamn är:", df.columns)
print("\nInformation om datan:", df.info())
print("\nData statistik:", df.describe(include="all"))
print("\nHela DataFrame: ", df.to_string()) #använd to_string()för att skriva ut hela DataFrame.
df.rename(columns=lambda x: x.strip(), inplace=True) # för att få bort mellanslag i kolumnnamn
print(df.columns)

# Transform: Rensa och bearbeta data:
# Kontrollera och hantera saknade värden

print("\nSaknade värden per kolumn före rensning:")
print(f"\nhasNaN")
hasNaN = df[df.columns].isna() # visa om värde är null, true och om inte false
print(hasNaN)

# eller andra metod:
print(df.isnull().sum()) # visa om värde är null, 1 och om inte 0

# Räknar antal kommuner
antal = df["Kommun"].count()
print(f"\nAntal kommun: ", antal)

# Filtrera data för att fokusera på kommuner med en skattesats över medelvärde
medel_skattesats_df = df["Summa, exkl. kyrkoavgift"].mean()
df['Genomsnitt'] = medel_skattesats_df
hoga_skatt_df = df[df["Summa, exkl. kyrkoavgift"] > medel_skattesats_df]
laga_skatt_df = df[df["Summa, exkl. kyrkoavgift"] < medel_skattesats_df]
print(f"\nMedelvärde av skattesats: ",medel_skattesats_df) # medelvärdet (33.43%)
print(f"\nHögst skattssats än Medelvärdet: ",hoga_skatt_df)#Antal län med skattesats över medelvärdet (33.43%): 20
print(f"\nLägst skattsats än Medelvärdet: ", laga_skatt_df)#Antal län med skattesats under medelvärdet (33.43%): 11

# Gruppindelning och aggregering för att hitta medelskattesatsen per kommun
medel_per_kommun_df = df.groupby("Kommun")["Summa, exkl. kyrkoavgift"].agg(["min","mean", "max"])
print(medel_per_kommun_df)

print("Data efter bearbetning:\n", df.tail(5))


# Utför utforskande dataanalys (EDA) med minst tre olika visualiseringar:
# histodiagram för fördelning av skattesats per kommuner
import seaborn as sns
plt.figure(figsize=[6,12])
sns.histplot(df["Summa, exkl. kyrkoavgift"], bins=7, kde=True, color="blue")
plt.title("SkattesatsFördelning per kommuner", fontsize=14, color="red")
plt.xlabel("Skattsats exkl. kyrkoavgift % ", fontsize=10, color="red")
plt.ylabel("Antal Kommuner", fontsize=10, color="red")
plt.xticks(rotation=45)
plt.show()
# Resultat: visar sig att flesta kommuner har skattesats mellan 33% - 34%

# Kontrollera och rensa kolumnnamn
df.columns = df.columns.str.strip()  # Tar bort extra mellanslag
df.columns = df.columns.str.lower()  # Omvandlar alla kolumnnamn till små bokstäver

# Gruppera församlingskod baserat på de två första siffrorna
df['församlings_kod_grupp'] = df['församlings-kod'].astype(str).str[:2]

# Aggregera skattesatsen per församlingskod
församling_agg = df.groupby('församlings-kod')['summa, exkl. kyrkoavgift'].agg(
    ['min', 'mean', 'max']).reset_index()
församling_agg.rename(columns={'min': 'Min', 'mean': 'Medel', 'max': 'Max'}, inplace=True)

# Skapa en dictionary för att mappa församlingskoder till län
lan_namn = {
    "01": "Stockholms län",
    "03": "Uppsala län",
    "04": "Södermanlands län",
    "05": "Östergötlands län",
    "06": "Jönköpings län",
    "07": "Kronobergs län",
    "08": "Kalmar län",
    "09": "Gotlands län",
    "10": "Blekinge län",
    "11": "Kristianstads län",
    "12": "Skåne län",
    "13": "Hallands län",
    "14": "Västra Götalands län",
    "15": "Älvsborgs län",
    "16": "Skaraborgs län",
    "17": "Värmlands län",
    "18": "Örebro län",
    "19": "Västmanlands län",
    "20": "Dalarnas län",
    "21": "Gävleborgs län",
    "22": "Västernorrlands län",
    "23": "Jämtlands län",
    "24": "Västerbottens län",
    "25": "Norrbottens län"
}

# Skapa en ny kolumn "Län" genom att mappa församlingskoden
df["Län"] = df["församlings-kod"].astype(str).str[:2].map(lan_namn)

# Slå ihop aggregerad data med län
df = df.merge(församling_agg, on='församlings-kod', how='left')


# Beräkna medelvärdet av skattesatsen
medel_skattesats = df["Medel"].mean()

# Filtrera data för län med skattesats över och under medelvärdet
over_medel = df[df["Medel"] > medel_skattesats]
under_medel = df[df["Medel"] < medel_skattesats]

# Räkna antalet län i varje kategori
antal_over_medel = len(over_medel["Län"].unique())
antal_under_medel = len(under_medel["Län"].unique())

# Skriv ut resultaten
print(f"Antal län med skattesats över medelvärdet ({medel_skattesats:.2f}%): {antal_over_medel}")
print(f"Antal län med skattesats under medelvärdet ({medel_skattesats:.2f}%): {antal_under_medel}")
# Stapeldiagram:
# Skapa en barplot för att visualisera antalet län i varje kategori
plt.figure(figsize=(10, 6))
plt.bar(["Över medelvärdet", "Under medelvärdet"], [antal_over_medel, antal_under_medel])
plt.xlabel("Antal län")
plt.ylabel("Antal län")
plt.title("Antal län i varje kategori")
plt.show()
#resultat: Antal län med skattesats över medelvärdet (34.66%): 20
           #Antal län med skattesats under medelvärdet (34.66%): 11


# Skapa en färgpalett för stapeldiagrammet
num_lan = df["Län"].nunique()  # Antal unika län
colors = plt.cm.get_cmap("tab20", num_lan)  # Färgpalett med olika färger

# Skapa ett stapeldiagram med olika färger för varje län
plt.figure(figsize=(12, 6))
bars = plt.bar(df["Län"], df["Medel"], color=[colors(i) for i in range(num_lan)], alpha=0.7)

# Titel och etiketter
plt.title("Medelvärde skattesats per län", fontsize=14)
plt.xlabel("Län", fontsize=12)
plt.ylabel("Medelvärde Skattsats % (Summa, exkl. kyrkoavgift)", fontsize=12)

# Rotera x-ticks för att de ska synas bättre
plt.xticks(rotation=90, fontsize=10, color="black")

# Tight layout för att undvika överlapp
plt.tight_layout()
plt.show()


# Skapa en ny DataFrame med aggregerad data per län
lan_agg = df.groupby('församlings_kod_grupp').agg(
    {'Län': 'first', 'Min': 'mean', 'Medel': 'mean', 'Max': 'mean'}
).reset_index()
lan_agg.rename(columns={'församlings_kod_grupp': 'Län_kod'}, inplace=True)

print(lan_agg.to_string(index=False))

# Spara csv.fil till excel.fil:

df.to_excel("Skattesats_per_kommuner2025.xlsx", sheet_name="kommun", index=False)

# Spara båda skattesats per kommuner och län på samma excel fil men på olika blad

with pd.ExcelWriter("Skattesats_per_kommuner2025.xlsx") as writer:
    df.to_excel(writer, sheet_name="Kommuner", index=False)
    lan_agg.to_excel(writer, sheet_name="Län", index=False)


#histogrammet för fördelningen av skattesats per län :
import seaborn as sns
plt.figure(figsize=[6,8])
sns.histplot(lan_agg["Medel"], bins=7, kde=True, color="blue")
plt.title("Fördelning av skattesats per län", fontsize=12, color="blue")
plt.xlabel("Medelvärde skattsats % (Summa, exkl. kyrkoavgift)", fontsize=10, color="red")
plt.ylabel("Antal län", fontsize=10, color="red")
plt.xticks(rotation=90)
plt.show() # output visa mest län har skattsats nästan 33.5% - 34.5% exkl krykoavgifter


#Linjediagram för att visa trender över tid.
# Skapa en ny DataFrame med aggregerad data per år
år_agg = df.groupby('år').agg({'summa, exkl. kyrkoavgift': 'mean'}).reset_index()

# Skapa ett linjediagram för att visa trender över tid
plt.figure(figsize=(12, 6))
plt.plot(år_agg['år'], år_agg['summa, exkl. kyrkoavgift'], marker='o', linestyle='-')
plt.xlabel("År", fontsize=12)
plt.ylabel("Medel Skattesats %", fontsize=12)
plt.title("Trend över tid för medel skattesats", fontsize=14)
plt.grid(True)  # Lägg till ett rutnät
plt.show() # output medel skattesats exkl. kyrkoavgift mindre än 33.5% nästan 33.4%

# Resultat: DataFrame har bara år 2025 så det går inte att visa trender över tid,
# Därför kommer jag att exportera medel_skattesats i sverige under år 2000-2025 genom ekonomifakta:
# https://www.ekonomifakta.se/sakomraden/skatt/skatt-pa-arbete/kommunalskatter_1212487.html

ar = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
medel_skattesats = [30.38, 30.53, 30.52, 31.17, 31.51, 31.60, 31.60, 31.55, 31.44, 31.52, 31.56, 31.55, 31.60, 31.73, 31.86, 31.99, 32.10, 32.12, 32.12, 32.19, 32.28, 32.27, 32.24, 32.24, 32.37, 32.41]
plt.figure(figsize=(14, 8), dpi=100)
plt.plot(ar,medel_skattesats, marker='o', linestyle='-', color="blue")
plt.xlabel("År", fontsize=12, color="blue")
plt.ylabel("Medel Skattesats % (exkl begravning och krykoavgifter)", fontsize=12, color="blue")
plt.title("Trend över tid för medel skattesats(exkl begravning och krykoavgifter) i Sverige ", fontsize=14, color="blue")
plt.tight_layout()
plt.grid(True)  # Lägg till ett rutnät
plt.show() # resultat : visar att skattesats har ökat 2.03 % från 2000 till år 2025


# Linjär diagram för medelvärde skattesats per kommuner
import matplotlib.pyplot as plt
plt.figure(figsize=(24, 8), dpi=300)
plt.plot(df["kommun"], df["summa, exkl. kyrkoavgift"], label=" Medelvärde Skattesats", alpha= 0.7, color="blue", linewidth=2, marker="o")
plt.title("Skattesats (summa, exkl. kyrkoavgift) per kommuner", fontsize=14, color="blue")
plt.xlabel("Kommuner", fontsize=8, alpha= 0.5,  color="red")
plt.ylabel("Medelvärde Skattesats %", fontsize=12, color="red")
plt.xticks(rotation=90, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Skattesats", fontsize=10)
plt.tight_layout()
plt.show() # resultat visar att det är svårt att läsa resultat eftersom det finns 289 kommuner
# därför kommer jag att visa Stockholm kommun

# Linjär diagram för enda Stockholm kommun med församlingar
# Konvertera kommunnamnet till versaler
import matplotlib.pyplot as plt
df_stockholm = df[df["kommun"].str.upper() == "STOCKHOLM"]
plt.figure(figsize=(12, 8))
plt.plot(df_stockholm["församling"], df_stockholm["summa, inkl. kyrkoavgift"], label="Skattesats", color="blue", linewidth=2, marker="o")
plt.title("Skattesats i Stockholms kommun per församling", fontsize=14, color="blue")
plt.xlabel("Församling", fontsize=12, color="red")
plt.ylabel("Skattesats % (inkl. kyrkoavgift)", fontsize=12, color="red")
plt.xticks(rotation=90, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Skattesats", fontsize=10)
plt.tight_layout()
plt.show() # resultat visar att Adolf fredriks församling har högst skattesats och Västermalms församling har längst skattesats i Stockholm


# Linjediagram för medelskattesats per län

# Skapa figur och linjediagram

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(lan_agg["Län"], lan_agg["Medel"], label="Medel", color="red", linewidth=2, marker="o")
plt.title("Genomsnittlig skattesats per län", fontsize=14, color="red")
plt.xlabel("Län", fontsize=12, color="red")
plt.ylabel("Skattesats %", fontsize=12, color="red")

plt.xticks(rotation=90, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Skattesats", fontsize=10)
plt.tight_layout()
plt.show() # output: Stockholm har längst skattesats och Västernorrlands län och Västerbotten län har högst skattesats


# Skapa scatter plot för att visa Skattesats per kommuner
plt.figure(figsize=(24, 8), dpi=300)
plt.scatter(df["kommun"], df["Medel"], color="purple", alpha=0.9, edgecolors="k")
plt.title("Scatter Plot: Skattesats per kommuner", fontsize=12, color="purple")
plt.xlabel("Kommuner", fontsize=12, alpha=0.9,  color="purple")
plt.ylabel("Skattesats %", fontsize=12, color="purple")
# Sätt ut kommunnamn var 10:e kommun för läsbarhet
plt.xticks(rotation=90, fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show() # output: det är svårt att läsa resultat
# därför kommer att Skapa scatter plot för att visa Skattesats per län


# Skapa scatter plot för att visa Skattesats per län
plt.figure(figsize=(12, 6), dpi=300)
plt.scatter(df["Län"], df["Medel"], color="purple", alpha=0.7, edgecolors="k")
plt.title("Scatter Plot: Skattesats per kommuner", fontsize=14, color="purple")
plt.xlabel("Kommuner", fontsize=12, color="purple")
plt.ylabel("Skattesats %", fontsize=12, color="purple")
plt.xticks(rotation=90, fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show() # output: det visar sig t.ex. att Stockholm har Medel skattesats % större än 29% mindre än 33%

"""
Det är viktigt för att skapa Scatter plot för att visa relationer mellan två variabler,
att veta hur förhållandet mellan x-axelns värden och y-axelns värden är, 
om det inte finns något samband kan den linjära regressionen inte användas för att förutsäga någonting.

Detta förhållande - korrelationskoefficienten - kallas r.
Värdet r sträcker sig från -1 till 1, där 0 betyder inget samband och 1 (och -1) betyder 100 % relaterat.

Python och Scipy-modulen kommer att beräkna detta värde åt dig, 
allt du behöver göra är att mata det med x- och y-värdena.

"""

from scipy import stats

x = df["summa, inkl. kyrkoavgift"]
y = df["summa, exkl. kyrkoavgift"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(f"\nkorrelationskoefficienten är: ", r) # output 0.9855686722972213 det betyder att finns relation

from scipy import stats
x = df["kommunal-skatt"]
y = df["landstings-skatt"]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(f"\nkorrelationskoefficienten är: ", r) #output -0.8933775631261186 det betyder att finns relation

from scipy import stats
x = df["begravnings-avgift"]
y = df["kyrkoavgift"]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(f"\nkorrelationskoefficienten är: ", r)# output 0.2320495990709025 det betyder att inte finns relation

# Korrelation Heatmap
df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelation mellan skattesatser och andra variabler")
plt.show() # resultat visar att relation mellan landstins-skatt och kommunal-skatt

