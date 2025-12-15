import pandas as pd
import numpy as np


df = pd.read_csv('CNT_Reinforced Concrete_Final Dataset.csv')

# Data Cleaning
df_clean = df.drop(columns=['S.No', 'Unnamed: 8'])
cols_to_numeric = ['Curing (days)', 'Cement (kg/m³)',
                   'Coarse aggregate (kg/m³)', 'CNT (%)']
for col in cols_to_numeric:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()


def augment_gaussian_noise(df, target_size=1000, noise_level=0.05):

    augmented_data = []
    stds = df.std()

    # how many new samples we need per row ko lagi
    current_size = len(df)
    n_new_samples_needed = target_size - current_size

    # Let's augment every row 'k' times such that total > target_size

    k = int(np.ceil(n_new_samples_needed / current_size))

    for _ in range(k):
        for _, row in df.iterrows():
            noise = np.random.normal(
                0, noise_level, size=len(row)) * stds.values
            new_row = row.values + noise
            new_row[new_row < 0] = 0  # Clip negative values
            augmented_data.append(new_row)

    # Combine garne
    columns = df.columns
    df_aug_only = pd.DataFrame(augmented_data, columns=columns)
    df_final = pd.concat([df, df_aug_only], ignore_index=True)

    return df_final


# augmentation
# Current size is 295. Target is 1000.
# 1000 - 295 = 705 needed.
# 705 / 295 = 2.38 -> ceil is 3.
# So we generate 3 new samples per row.
# Total will be 295 + (295 * 3) = 1180.
df_augmented = augment_gaussian_noise(
    df_clean, target_size=1000, noise_level=0.05)


df_augmented = df_augmented.sample(frac=1).reset_index(drop=True)

print(f"Original size: {len(df_clean)}")
print(f"Augmented size: {len(df_augmented)}")

df_augmented.to_csv('CNT_Concrete_Augmented_1000.csv', index=False)


print(df_augmented.head().to_markdown(
    index=False, numalign="left", stralign="left"))
