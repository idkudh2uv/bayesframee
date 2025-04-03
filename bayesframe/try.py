#import module
import pandas as pd
from linreg import LinReg

# Load the data from CSV
data_path = "data\data.csv"
df = pd.read_csv(data_path)  # Read CSV with headers

# Extract target (y) and features (X)
y = df["Target"].values  # Target column
X = df.drop(columns=["Target"]).values  # All other columns as features

# Initialize the model
lin = LinReg(X=X, y=y, val_scheme="leave_one_out")

# Print model data
print(lin.get_model_data())
