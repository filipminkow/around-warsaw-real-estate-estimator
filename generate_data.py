import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

data = {
    'metraz': np.random.uniform(35, 100, n_samples).round(1),
    'pokoje': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.3, 0.2]),
    'wiek_budynku': np.random.randint(0, 50, n_samples),
    'odleglosc_pkp_km': np.random.uniform(0.1, 5.0, n_samples).round(2),
    'miasto': np.random.choice(['Minsk Mazowiecki', 'Grodzisk Mazowiecki', 'Blonie', 'Pruszkow', 'Piaseczno'], n_samples)
}

df = pd.DataFrame(data)

# Mapa dopłat za lokalizację (im bliżej i lepszy dojazd, tym drożej)
doplaty_miasta = {
    'Piaseczno': 140000,
    'Pruszkow': 120000,
    'Grodzisk Mazowiecki': 70000,
    'Minsk Mazowiecki': 40000,
    'Blonie': 20000
}

cena_bazowa = 350000
df['cena_pln'] = (
    cena_bazowa 
    + df['miasto'].map(doplaty_miasta)
    + (df['metraz'] * 6000) 
    + (df['pokoje'] * 20000) 
    - (df['wiek_budynku'] * 2500) 
    - (df['odleglosc_pkp_km'] * 30000)
    + np.random.normal(0, 30000, n_samples)
).round(0)

df.to_csv('mieszkania_okolice_warszawy.csv', index=False)
print("Gotowe! Nowy zbiór danych z uwzględnieniem cen miast został zapisany.")