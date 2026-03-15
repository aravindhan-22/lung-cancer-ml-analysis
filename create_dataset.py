import pandas as pd
import numpy as np

np.random.seed(42)

data = pd.DataFrame({
    "AGE": np.random.randint(30,80,300),
    "SMOKING": np.random.randint(0,2,300),
    "YELLOW_FINGERS": np.random.randint(0,2,300),
    "ANXIETY": np.random.randint(0,2,300),
    "PEER_PRESSURE": np.random.randint(0,2,300),
    "CHRONIC_DISEASE": np.random.randint(0,2,300),
    "FATIGUE": np.random.randint(0,2,300),
    "ALLERGY": np.random.randint(0,2,300),
    "WHEEZING": np.random.randint(0,2,300),
    "ALCOHOL_CONSUMING": np.random.randint(0,2,300),
    "COUGHING": np.random.randint(0,2,300),
    "SHORTNESS_OF_BREATH": np.random.randint(0,2,300),
    "CHEST_PAIN": np.random.randint(0,2,300),
    "LUNG_CANCER": np.random.randint(0,2,300)
})

data.to_csv("data/lung_cancer.csv",index=False)

print("Dataset created successfully!")