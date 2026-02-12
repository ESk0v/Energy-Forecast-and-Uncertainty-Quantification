from MainLibrary import GetData, SavePlot
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import dates
import pathlib
import sys

Data = GetData() #indsæt csv fil navn

Data = Data[["dateTime","toutdoor","treturn","tforward"]]
Data = Data.sort_values("dateTime").set_index("dateTime")

Data.index = pd.to_datetime(Data.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(Data.index, df["treturn"], label="Temperature In", color="steelblue") #placeholder navne, indsæt rigtig kolonne navne
ax.plot(Data.index, df["tforward"], label="Temperature Out", color="crimson") #placeholder navne, indsæt rigtig kolonne navne
ax.plot(Data.index, df["toutdoor"], label="Temperature Outside", color="seagreen") #placeholder navne, indsæt rigtig kolonne navne

ax.set_xlabel("Date (YYYY-MM-DD)")
ax.set_ylabel("Temperature(°C)")
ax.set_title("Temperature over Date (Hourly)")
ax.legend()

plt.tight_layout()
SavePlot(fig) #en bestemt fil den skal gemmes i
plt.show()
