import pandas as pd
from tgan.model import TGANModel

df = pd.read_csv('../CNT_Reinforced Concrete_Final Dataset.csv')    

target_col = 'Compressive strength (MPa)'        
continuous_cols = list(df.columns)       

tgan = TGANModel(
    continuous_columns=continuous_cols,
    output_column=target_col,
    max_epoch=5,          # increase for better quality
    steps_per_epoch=100,  # "
    batch_size=32
)

tgan.fit(df)

n_synthetic_rows = len(df)                # to generate synthetic data of same size as original
synth = tgan.generate(n_synthetic_rows)

synth.to_csv('../augmented_tgan.csv', index=False)

print('Saved csv')
