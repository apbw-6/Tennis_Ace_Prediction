# my_libraries.py - A central file for common imports

# Data handling
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

# Joblib for saving/loading models
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Other utilities
import tennis_utils
import dataframe_utils

# Display all columns
pd.set_option("display.max_columns", None)
# Palette
sns.set_palette("colorblind")  # Default Seaborn colors

# Constants
baseline_x = 11.885
service_line_x = 6.4
singles_sideline_y = 4.115