#Uppgift 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import alpha

# Ställ in Pandas för att visa fler rader
pd.options.display.max_rows = 1500

# Läs in CSV-filen
df = pd.read_csv("skattsats2025.csv", encoding="utf-8-sig", sep=";", decimal=".", skipinitialspace=True, low_memory=False)

df.rename(columns=lambda x: x.strip(), inplace=True) # för att få bort mellanslag i kolumnnamn
print(df.columns)


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


#Spara båda skattesats per kommuner och län på samma excel fil men på olika blad

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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Rensa data genom att ta bort rader med NaN-värden
df = df.dropna(subset=["Medel"])

# Utför K-means klustring med 3 kluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Medel"]])

# Skapa färgskala baserad på skattesats
df = df.sort_values(by="kommun")  # Behåller ordningen på kommunerna

# Bestäm färgerna baserat på skattesatsen
colors = np.where(df["Medel"] < 31, "yellow",  # Låg skatt
                 np.where(df["Medel"] < 33, "orange", "red"))  # Medel skatt & Hög skatt

# Skapa scatter plot för att visa skattesats per kommun
plt.figure(figsize=(24, 8), dpi=300)

plt.scatter(df["kommun"], df["Medel"], color=colors, alpha=0.9, edgecolors="k")

plt.title("Scatter Plot: Skattesats per kommuner med färgkodning", fontsize=14, color="purple")
plt.xlabel("Kommuner", fontsize=12, color="purple")
plt.ylabel("Skattesats %", fontsize=12, color="purple")

# Visa alla kommuner i rätt ordning
plt.xticks(rotation=90, fontsize=8)

plt.grid(True, linestyle="--", alpha=0.6)

# Lägg till en legend manuellt
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color="yellow", label="Låg skatt (<31%)"),
                  mpatches.Patch(color="orange", label="Medel skatt (31-33%)"),
                  mpatches.Patch(color="red", label="Hög skatt (>33%)")]

plt.legend(handles=legend_patches, loc="upper right")
plt.tight_layout()
plt.show()


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

#Klustring: Identifiera grupper (regioner: län ).
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Rensa data genom att ta bort rader med NaN-värden eftersom K-means fungerar inte med saknade värden
df = df.dropna(subset=["Medel"])

# Utför K-means klustring
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Specificera n_init för att undvika varningar
df["Cluster"] = kmeans.fit_predict(df[["Medel"]]) # Tilldelar varje kommun ett klusternummer (0 eller 1)

# Ge klustren beskrivande namn
cluster_names = {1: "Låg skatt", 0: "Hög skatt"}
df["Cluster_namn"] = df["Cluster"].map(cluster_names)

# Färgkodning för kluster
cluster_colors = {"Låg skatt": "yellow", "Hög skatt": "red"}

# Skapa scatter plot för att visa Skattesats per län med klusterfärger
plt.figure(figsize=(12, 6), dpi=300)

for cluster, color in cluster_colors.items():
    cluster_data = df[df["Cluster_namn"] == cluster]
    plt.scatter(cluster_data["Län"], cluster_data["Medel"],
                color=color, alpha=0.7, edgecolors="k", label=cluster)

plt.title("Scatter Plot: Skattesats per län med kluster", fontsize=14, color="purple")
plt.xlabel("Län", fontsize=12, color="purple")
plt.ylabel("Skattesats %", fontsize=12, color="purple")
plt.xticks(rotation=90, fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


"""
Det är viktigt för att skapa Scatter plot för att visa relationer mellan två variabler,
att veta hur förhållandet mellan x-axelns värden och y-axelns värden är, 
om det inte finns något samband kan den linjära regressionen inte användas för att förutsäga någonting.

Detta förhållande - korrelationskoefficienten - kallas r.
Värdet r sträcker sig från -1 till 1, där 0 betyder inget samband och 1 (och -1) betyder 100 % relaterat.

Python och Scipy-modulen kommer att beräkna detta värde åt dig, 
allt du behöver göra är att mata det med x- och y-värdena.

"""

# Korrelation Heatmap
df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelation mellan skattesatser och andra variabler")
plt.show() # resultat visar att relation mellan landstins-skatt och kommunal-skatt


# uppgift 3

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import ace_tools_open as tools  # För att visa prognosen som tabell

# Läs in filen
filnamn = "Skattesats_per_kommuner2025 3.xlsx"
df = pd.read_excel(filnamn, sheet_name="År")

# Identifiera skattekolumner
skatt_kolumner = [col for col in df.columns if "Skatt" in col]

# Omvandla data till långt format
df_long = df.melt(id_vars=["Kommun"], value_vars=skatt_kolumner, var_name="År", value_name="Skattesats")

# Extrahera årtal
df_long["År"] = df_long["År"].str.extract("(\d+)").astype(int)

# Ta bort NaN-värden
df_long = df_long.dropna(subset=["Skattesats"])

# Konvertera skattesatsen till numerisk
df_long["Skattesats"] = pd.to_numeric(df_long["Skattesats"], errors="coerce")

# Gruppera data efter år och beräkna medelskattesatsen
df_grouped = df_long.groupby("År")["Skattesats"].mean().reset_index()

# Regression
X = df_grouped["År"].values.reshape(-1, 1)
y = df_grouped["Skattesats"].values

model = LinearRegression()
model.fit(X, y)

# Prognos för framtida år
framtida_ar = np.array([2026, 2027, 2028]).reshape(-1, 1)
prognoser = model.predict(framtida_ar)

# Skapa DataFrame för prognoser
df_prognos = pd.DataFrame({"År": framtida_ar.flatten(), "Förväntad Skattesats": prognoser})

# Visa prognoser i en tabell
tools.display_dataframe_to_user(name="Förutsägelse av skattesats", dataframe=df_prognos)

# Visualisera resultat
plt.figure(figsize=(10, 5))
plt.plot(df_grouped["År"], df_grouped["Skattesats"], marker="o", label="Historisk data")
plt.plot(framtida_ar, prognoser, marker="x", linestyle="--", color="red", label="Prognos")

# Diagraminställningar (utan emoji för att undvika fontproblem)
plt.xlabel("År")
plt.ylabel("Genomsnittlig skattesats (%)")
plt.title("Prognos av kommunal skattesats")  # Ändrad titel för att undvika emoji-problem
plt.legend()
plt.grid(True)
plt.show()


