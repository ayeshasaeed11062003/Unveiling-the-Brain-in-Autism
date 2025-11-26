import pandas as pd

df = pd.read_csv(
    "C:/Users/ayesh/OneDrive/Desktop/Unveiling-the-Brain-in-Autism/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
)

print(df.head(20))
print(df[["SUB_ID", "SITE_ID"]].head(20))
