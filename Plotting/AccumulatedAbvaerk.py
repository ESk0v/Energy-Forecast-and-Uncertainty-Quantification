import pandas as pd
import matplotlib.pyplot as plt
#from MainLibrary import GetData, SavePlot

# Load CSV
df = pd.read_csv(
    "",
    sep=";",
    decimal=",",
    low_memory=False
)

# Convert ID column to datetime
df["ID"] = pd.to_datetime(df["ID"], format="%d/%m/%Y %H.%M")

# Ensure abvaerk is numeric
df["abvaerk"] = pd.to_numeric(df["abvaerk"], errors="coerce")

# Create a date-only column
df["date"] = df["ID"].dt.date

# Group by day and sum
daily_abvaerk = df.groupby("date")["abvaerk"].sum()

# Plot
plt.figure()
plt.plot(daily_abvaerk.index, daily_abvaerk.values)

plt.xlabel("Date")
plt.ylabel("Accumulated abvaerk (hours)")
plt.title("Daily Accumulated Abvaerk")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
