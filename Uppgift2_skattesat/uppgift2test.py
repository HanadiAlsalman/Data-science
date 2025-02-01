import pandas as pd
import matplotlib.pyplot as plt

# Kontrollera antalet maximalt returnerade rader:
pd.options.display.max_rows = 1500

# Extract: Hämta och importera csv_fil och läsa in data:
df = pd.read_csv("/Users/hanadialsalman/Documents/Data-science/Uppgift2_skattesat/skattsats2025.csv", header=0, sep= ";", decimal=".", skipinitialspace=True, low_memory=False)
# df är en variabel som innehåller data från dataportal.se
#encoding='latin1':Många CSV-filer från europeiska källor är kodade med latin1(alt. kan du testa encoding='cp1252')
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
#df.rename(columns=lambda x: x.strip(), inplace=True) # # för att få bort mellanslag i kolumnnamn
print(df.columns)

# Transform: Rensa och bearbeta data:
# Kontrollera och hantera saknade värden

print("\nSaknade värden per kolumn före rensning:")

print(f"\nhasNaN")
hasNaN = df[df.columns].isna() # visa om värde är null, true och om inte false
print(hasNaN)

# eller andra metod:
print(df.isnull().sum()) # visa om värde är null, 1 och om inte 0


# Skapa nya kolumner baserat på beräkningar eller villkor.
df["Procent_landsting_i_total"] = df["Landstings-skatt"] / df["Summa, exkl. kyrkoavgift"] * 100 # för att få procent av landstingskatt av totalskatt
print(df.head(5))

# Räknar antal kommuner
antal = df["Kommun"].count()
print(f"\nAntal kommun: ", antal) # output Antal kommun:  1282

# Filtrera data för att fokusera på kommuner med en skattesats över rikets medelvärde
medel_skattesats_df = df["Summa, exkl. kyrkoavgift"].mean()
medel_skattesats_utan_begravning_df = (df["Summa, exkl. kyrkoavgift"]-df["Begravnings-avgift"]).mean()
hoga_skatt_df = df[df["Summa, exkl. kyrkoavgift"] > medel_skattesats_df]
laga_skatt_df = df[df["Summa, exkl. kyrkoavgift"] < medel_skattesats_df]

print(f"\nMedelvärde av skattesats exkl krykoavgifter: ",medel_skattesats_df) # output Medelvärde av skattesats:  33.43471762870515
print(f"\nMedelvärde av skattesats exkl båda begravning och krykoavgifter: ",medel_skattesats_utan_begravning_df) # 33.146630265210604
print(f"\nHögst skattssats än Medelvärdet: ",hoga_skatt_df)
print(f"\nLägst skattsats än Medelvärdet: ", laga_skatt_df)

# Gruppindelning och aggregering för att hitta medelskattesatsen per kommun
medel_per_kommun_df = df.groupby("Kommun")["Summa, exkl. kyrkoavgift"].agg(['mean']).reset_index()
print(medel_per_kommun_df)

df["Medel"] = medel_per_kommun_df["mean"] # skapa ny kolumn för genomsnitt skattsats per kolumn

print("Data efter bearbetning:\n", df.head(5))

# Utför utforskande dataanalys (EDA) med minst tre olika visualiseringar:
# histodiagram för fördelning av skattesats per kommuner
import seaborn as sns
plt.figure(figsize=(6, 8))
sns.histplot(df["Medel"], bins=7, kde=True, color="blue")
plt.title("Fördelning av skattesats per kommuner", fontsize=14, color="red")
plt.xlabel("Medelvärde skattsats % (Summa, exkl. kyrkoavgift)", fontsize=10, color="red")
plt.ylabel("Antal Kommuner", fontsize=10, color="red")
plt.xticks(rotation=45)
plt.show()
# Resultat: Histogrammet visade att de flesta kommuner har en total kommunal skattesats (exkl kyrkoavgift) mellan 34%

# Skapa ett linjediagram för att visa trender över tid
plt.figure(figsize=(12, 6))
plt.plot(df['År'], df['Summa, exkl. kyrkoavgift'], marker='o', linestyle='-')
plt.xlabel("År", fontsize=12)
plt.ylabel("kommunal skattesats (exkl kyrkoavgift) %", fontsize=12)
plt.title("Trend över tid för kommunal skattesats (inkl kyrkoavgift)", fontsize=14)
plt.grid(True)  # Lägg till ett rutnät
plt.show()

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
plt.figure(figsize=(12, 8))
plt.plot(df["Kommun"], df["Medel"], label=" Medelvärde Skattesats", color="blue", linewidth=2, marker="o")
plt.title("medelvärde skattesats per kommuner", fontsize=14, color="blue")
plt.xlabel("Kommuner", fontsize=12, color="red")
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
df_stockholm = df[df["Kommun"].str.upper() == "STOCKHOLM"]
plt.figure(figsize=(12, 8))
plt.plot(df_stockholm["Församling"], df_stockholm["Summa, inkl. kyrkoavgift"], label="Skattesats", color="blue", linewidth=2, marker="o")
plt.title("Skattesats i Stockholms kommun per församling", fontsize=14, color="blue")
plt.xlabel("Församling", fontsize=12, color="red")
plt.ylabel("Skattesats % (inkl. kyrkoavgift)", fontsize=12, color="red")
plt.xticks(rotation=90, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Skattesats", fontsize=10)
plt.tight_layout()
plt.show() # resultat visar att Adolf fredriks församling har högst skattesats och Västermalms församling har längst skattesats i Stockholm


#Scatter plot för att visa relationer mellan två variabler.

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

x = df["Summa, inkl. kyrkoavgift"]
y = df["Summa, exkl. kyrkoavgift"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(f"\nkorrelationskoefficienten är: ", r) # output 0.9855686722972213 det betyder att finns relation

from scipy import stats

x = df["Kommunal-skatt"]
y = df["Landstings-skatt"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(f"\nkorrelationskoefficienten är: ", r) #output -0.8933775631261186 det betyder att finns relation

from scipy import stats

x = df["Begravnings-avgift"]
y = df["Kyrkoavgift"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(f"\nkorrelationskoefficienten är: ", r)# output 0.2320495990709025 det betyder att inte finns relation

import numpy as np
# Korrelation Heatmap
df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelation mellan skattesatser och andra variabler")
plt.tight_layout( )
plt.show() # resultat visar att relation mellan landstins-skatt och kommunal-skatt

# Scatter plot för att visa hur mycket procent landstings-skatt av total kommunal skatt per kommuner

import matplotlib.pyplot as plt

# Skapa en större och tydligare figur för att rymma alla kommuner
plt.figure(figsize=(24, 8), dpi=200)
plt.scatter(df["Kommun"], df["Procent_landsting_i_total"], alpha=0.7, color="hotpink", label="Landstings-skatt")
plt.title("hur mycket procent landstings-skatt av total kommunal skatt per kommuner", fontsize=16, fontweight='bold')
plt.xlabel("Kommun", fontsize=8, alpha=0.7, color="black")
plt.ylabel("Procent_landsting_i_totalskatt %", fontsize=12,alpha=0.7, color="black")
plt.xticks(rotation=90, fontsize=6, color="black")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout( )
plt.legend()
plt.show() # resultat visar att alla kommuner har landstings skatt bara kommun Gotland har inte landstings skattASDa


# Scatter plot för att visa Relation mellan landstins-skatt och kommunal-skatt
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(12, 6))
plt.scatter(df["Procent_landsting_i_total"], df["Landstings-skatt"], alpha=0.7, color="#88c999")
plt.title("Relation mellan landstins-skatt och kommunal-skatt")
plt.xlabel("Landstings-skatt %", fontsize=12, alpha=0.7, color="black")
plt.ylabel("Procent_landsting_i_total skatt", fontsize=12, alpha=0.7, color="black")
plt.xticks(rotation=90, fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show() # resultat visar att landstings skatt är nästan mellan 30% - 40% av total kommunal skatt

print(df.to_string(index=False))


df.to_excel("Skattesats_per_kommuner2025.xlsx", sheet_name="Kommun")
print("\nData har sparats till Excel.")
print(df.tail())


