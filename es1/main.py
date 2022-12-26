import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Data import
df = pd.read_csv("seriefit2021.csv")
x = df.x
y = df.y

# Show data on graph
plt.plot(x, y, label="Fit")
plt.title("Raw data")
plt.xlabel("Month")
plt.ylabel("Value")
plt.legend()
plt.show()

# Trend function
y2 = y.copy()
x2 = x.copy()

# Approssimazione polinomiale grado 2
poly2 = np.polyfit(x2, y2, 2)
p2 = np.poly1d(poly2)

# Grado 3
poly3 = np.polyfit(x2, y2, 3)
p3 = np.poly1d(poly3)

# Grado 4
poly4 = np.polyfit(x2, y2, 4)
p4 = np.poly1d(poly4)

# Grado 5
poly5 = np.polyfit(x2, y2, 5)
p5 = np.poly1d(poly5)

# Grado 6
poly6 = np.polyfit(x2, y2, 6)
p6 = np.poly1d(poly6)

# Grado 7
poly7 = np.polyfit(x2, y2, 7)
p7 = np.poly1d(poly7)

# Grado 8
poly8 = np.polyfit(x2, y2, 8)
p8 = np.poly1d(poly8)

# Grado 9
poly9 = np.polyfit(x2, y2, 9)
p9 = np.poly1d(poly9)

# Grado 10
poly10 = np.polyfit(x2, y2, 10)
p10 = np.poly1d(poly10)

plt.plot(x, y, label="Fit")
plt.plot(x2, p2(x2), label="PG2")
plt.plot(x2, p3(x2), label="PG3")
plt.plot(x2, p4(x2), label="PG4")
plt.plot(x2, p5(x2), label="PG5")
plt.plot(x2, p6(x2), label="PG6")
plt.plot(x2, p7(x2), label="PG7")
plt.plot(x2, p8(x2), label="PG8")
plt.plot(x2, p9(x2), label="PG8")
plt.plot(x2, p10(x2), label="PG10")

plt.legend()
plt.show()

# Osservando l'andamento delle funzioni, ritengo che il polinomio di grado 3 sia la migliore.
# Anche il polinomio di grado 2 potrebbe andare bene in quanto ha una curva simile/uguale a quella di grado 3.
plt.plot(x, y, label="Fit")
plt.title("Funzione approssimata")
plt.xlabel("Month")
plt.ylabel("Value")
plt.plot(x2, p3(x2), label="PG3")
plt.plot(x2, p2(x2), label="PG2")
plt.legend()
plt.show()

# Analizzo le componenti di trend, stagionalità e residuo
ds = df[df.columns[1]]  # Convert to series. Perché il decompose accetta solo una data series e non una colonna di
# data frame.

# Con il tipo di elongazione iniziale dei massimi e minimi e quelli che abbiamo verso la fine, allora possiamo
# dire che quello moltiplicativo è il modello migliore per eliminare l'effetto del trend. Questo perché
# il modello moltiplicativo moltiplica le elongazioni per il valore del trend,
# quindi se ci sono valori piccoli allora può moltiplicare per un valore piccolo e viceversa.
# Nel nostro caso le variazioni stagioni Sono piccoli in corrispondenza di dati piccoli e viceversa.Quindi il modello
# Moltiplicativo risulta essere la scelta migliore.

# Il modello additivo farebbe fatica a catturare questa variazione dell'elongazione in quanto
# aggiunge una certa quantità. E se la quantità è uguale aggiunge una quantità sempre uguale e quindi si avrebbero
# degli scostamenti sempre uguali.

result_additive = seasonal_decompose(ds, model='additive', period=12)
result_additive.plot()
plt.title("Additive model")
plt.show()

# I nostri dati sono aggregati per mese e quindi valutiamo su un period di 1 anno, quindi 12 mesi.
result_multiplicative = seasonal_decompose(ds, model='multiplicative', period=12)
result_multiplicative.plot()
plt.title("Multiplicative model")
plt.show()

# Trend
plt.plot(x, result_multiplicative.trend, color="blue", label="Trend")

# Detrend
# Prendo i dati originali e rimuovo il trend
detrend = y - result_multiplicative.trend
plt.plot(x, detrend, color="red", label="Detrend")
plt.show()

residue = result_multiplicative.resid
seasonal = result_multiplicative.seasonal

# ACF - Auto - correlation function
diff_data = ds.diff()
diff_data[0] = ds[0]  # reset 1st elem
sm.graphics.tsa.plot_acf(diff_data, lags=36)
plt.show()

# residuo e stagionalità
deseasonalized = y / result_multiplicative.seasonal
plt.plot(x, deseasonalized, color="red")
plt.legend("Deseasonalized")
plt.show()

# Dall'ACF possiamo dedurre che è già destagionalizzato.
# Confronto tra stagionalità calcolata attraverso "seasonal_decompose" e quella attraverso calcoli manuali
test = y - p3(x)
plt.plot(x, test)
plt.plot(x, detrend)
plt.title("Detrend calcolato / ottenuto")
plt.legend(["Calcolato", "Ottenuto"])
plt.show()

# Confronto fra trend originale e funzione di trend
plt.plot(x, result_multiplicative.trend)
plt.plot(x, p3(x))
plt.show()

# Predizione stagionalità
season_coeff = []
for d in [0, 1, 2, 3, 4, 5, 6]:
    season_coeff.append(np.mean(test[d::7]))

final_season = season_coeff[1:]
final_season = np.append(final_season, season_coeff)
final_season = np.resize(final_season, 36)

# Predizione per ulteriori 36 periodi di tempo
x1 = np.linspace(65, 101, 36)
x2 = np.linspace(65, 101, 36)
predictions_poly3 = (p3(x1)) + final_season

plt.plot(x, y)
plt.plot(x1, predictions_poly3)
plt.show()
