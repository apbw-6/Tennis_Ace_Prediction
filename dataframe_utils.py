# Import everything from my_libraries.py
from my_libraries import *

# Constants
baseline_x = 11.885
service_line_x = 6.4
singles_sideline_y = 4.115
tolerance = 0.25  # Setting tolerance (in meters) for new features

#############################################################################################
# DATA CLEANING
#############################################################################################

def clean_dataframe(df):
    """
    - Turn object datatype to string
    - Drop duplicates.
    - z-coordinate cannot be negative.
    - It is not possible that the server and the receiver are on the same side of the court. These points must be deleted.
    - It is not possible that the server and the ball bounce is on the same side of the court. These points must be deleted.
    - <b>Fliping the court</b> to make things simpler. I.e the server will always be on the left and the returner will always be on the right. This shouldn't make any difference to the game.
    - Server position must be close to the baseline and cannot be way inside the court or way behind.
    - Returner position must be near or behind the baseline and cannot be way inside the court.
    - Fastest serve speed recorded is 163.7mph. Serve speed cannot exceed this.
    - ball_net_z of 1.5m or above will be deleted. It's not probable that serves that are in go too high above the net. And height should be greater than 0.915m
    - - Delete outliers based on Inter Quartile Range of training data.
    
    Args:
        datafarme: to be cleaned

    Returns:
        dataframe: cleaned data
    """
    
    # Tennis court dimensions
    baseline_x = 11.885
    service_line_x = 6.4
    singles_sideline_y = 4.115 
    
    # Changing object data type to string in the dataframe
    obj_features = df.select_dtypes(include='object').columns
    df[obj_features] = df[obj_features].fillna('').astype(str)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # z-coordinates cannot be negative at all.
    df = df[(df["ball_hit_z"] >= 0) & (df["ball_net_z"] >= 0)]
    
    # It is not possible that the server and the receiver are on the same side of the court. These points must be deleted.
    df = df[
        ((df["ball_hit_x"] < 0) & (df["receiver_x"] > 0))
        | ((df["ball_hit_x"] > 0) & (df["receiver_x"] < 0))
    ]   
    
    # It is not possible that the server and the ball bounce is on the same side of the court. These points must be deleted.
    df = df[
        ((df["ball_bounce_x"] < 0) & (df["hitter_x"] > 0))
        | ((df["ball_bounce_x"] > 0) & (df["hitter_x"] < 0))
    ]
    
    # I will also flip the courts to make things simpler. Ie the server will always be on the left and the returner will always be on the
    # right. This shouldn't make any difference to the game.

    # If ball hit is on the right, ball_hit_x is positive.
    # Change sign of ball_hit_x and ball_hit_y to retain ad/deuce court.
    # Change sign of ball_net_y
    # Change sign of ball_bounce_x, ball_bounce_y
    # Change sign of hitter_x, hitter_y
    # Change sign of receiver_x, receiver_y

    # Multiply the column to be flipped by -1 if 'ball_hit_x' is positive.
    cols_to_flip = [
        "ball_hit_x",
        "ball_hit_y",
        "ball_net_y",
        "ball_bounce_x",
        "ball_bounce_y",
        "hitter_x",
        "hitter_y",
        "receiver_x",
        "receiver_y",
    ]
    df.loc[df["ball_hit_x"] > 0, cols_to_flip] *= -1   
    
    # Ball placement has to be inside the service box and not outside.
    # Adding 10cm tolerance due to thickness of the line and ball slippage.
    df = df[(df["ball_bounce_x"] <= service_line_x + 0.1)]
    df = df[
        (df["ball_bounce_y"] >= -singles_sideline_y - 0.1)
        & (df["ball_bounce_y"] <= singles_sideline_y + 0.1)
    ]
    
    # Server position must be close to the baseline and cannot be way inside the court or way behind.
    # Allowing 1 m behind baseline (generous) and 0.5m within baesline, and 1m within sidelines.
    df = df[(df["hitter_x"] >= -baseline_x - 1)]
    df = df[
        (df["hitter_y"] >= -singles_sideline_y + 1)
        & (df["hitter_y"] <= singles_sideline_y - 1)
    ]
    
    # Returner position must be near or behind the baseline and cannot be way inside the court.
    # Allowing 7 m behind baseline (generous) and 2m within baesline, and 1.5m within sidelines.
    df = df[(df["receiver_x"] <= baseline_x + 7) & (df["receiver_x"] >= baseline_x - 2)]
    df = df[
        (df["receiver_y"] >= -singles_sideline_y - 1.5)
        & (df["receiver_y"] <= singles_sideline_y + 1.5)
    ]
    
    # Fastest serve speed recorded is 163.7mph. Serve speed cannot exceed this.
    df = df[df["ball_hit_v"] <= 163.7]
    
    # ball_net_z of 1.5m or above will be deleted. It's not probable that serves that are in go too high above the net. 
    # And height should be greater than 0.915m
    df = df[(df["ball_net_z"] < 1.5) & (df["ball_net_z"] >= 0.915)]
    
    # It is not possible for ball speed to be greater at the net than when the server has just hit the ball. These points must be deleted.
    df = df[df["ball_hit_v"] > df["ball_net_v"]]   
    
    df.to_csv("datasets/train_dataset.csv")
    
    # Deleting points outside IQR bounds
    columns = [
        'ball_hit_y', 'ball_hit_x',
        'ball_hit_z', 'ball_hit_v', 'ball_net_v', 'ball_net_z', 'ball_net_y',
        'ball_bounce_x', 'ball_bounce_y', 'ball_bounce_v', 'ball_bounce_angle',
        'hitter_x', 'hitter_y', 'receiver_x', 'receiver_y'
    ]
    train_df = pd.read_csv('datasets/train_dataset.csv')
    # Compute IQR using TRAINING data
    Q1_train = train_df[columns].quantile(0.25)
    Q3_train = train_df[columns].quantile(0.75)
    IQR_train = Q3_train - Q1_train

    # Define bounds using training data
    lower_bound = Q1_train - 1.5 * IQR_train
    upper_bound = Q3_train + 1.5 * IQR_train
    
    for col in columns:
        df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])] 
    
    return df

def impute_data(df):
    """ - Impute missing data with mean and most frequent. """
    
    # Create an imputer that fills missing values with the most frequent category
    imputer = SimpleImputer(strategy="most_frequent")

    cols_most_frequent = [
        "surface",
        "serve_side",
        "serve_number",
        "hitter_hand",
        "receiver_hand",
    ]
    df[cols_most_frequent] = imputer.fit_transform(df[cols_most_frequent])
    
    # Impute missing values with column mean (only for numeric columns)
    # Note that serve_number will already be imputed before this.
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df


#############################################################################################
# FEATURE ENGINEERING
#############################################################################################

def feature_engineer(df):
    """
    - Perform one hot encoding for categorical features with the dummy variable trap in mind. 
    - Add new features:
        First I want to measure the distance between ball bounce and where the returner is positioned in three ways:
        - x-displacement between ball bounce and returner's position: if this distance is large then returner has to move more 
            along the baseline to hit the ball
        - y-displacement between ball bounce and returner's position
        - total distance between ball bounce and returner's position

        Next I want to measure how good a serve is by seeing how close it is to the lines: using arbitrary measure of closeness
        as 10cm (setting tolerence = 10cm). If close, the value is 1, else 0.
        - close_to_side_line
        - close_to_center_line: ie down the T
        - close_to_service_line
        
        I want to also bin some data and delete the old columns
        - ball_hit_v, ..., ...
        
    - Scale the data: Use both Robust and Standard scaling as the features have different distributions.
    - Use the correlation matrix to see which features have high positive or negative correlation. Use domain knowledge to drop
        or preserve features. Using a threshold of 0.9 (absolute value).  

    Args:
        datafarme: for feature engineering

    Returns:
        dataframe: feature engineering completed
    """
    
    #----------------------------------------------------------------------------------------
    # One hot encoding 
    #----------------------------------------------------------------------------------------
    # Load encoder
    loaded_encoder = joblib.load("one_hot_encoder.pkl")
    # Transform data
    processed_data = loaded_encoder.transform(df)
    # Get clean feature names by removing 'cat__' and 'remainder__'
    original_feature_names = loaded_encoder.get_feature_names_out()
    clean_feature_names = [name.split("__")[-1] for name in original_feature_names]
    
    df = pd.DataFrame(processed_data, columns=clean_feature_names, dtype=float)
    
    #----------------------------------------------------------------------------------------
    # New features
    #----------------------------------------------------------------------------------------
    df["dist_ball_bounce_x_returner_x"] = abs(df["ball_bounce_x"] - df["receiver_x"])
    df["dist_ball_bounce_y_returner_y"] = abs(df["ball_bounce_y"] - df["receiver_y"])
    df["dist_ball_bounce_returner_total"] = np.sqrt(
        df["dist_ball_bounce_x_returner_x"] ** 2 + df["dist_ball_bounce_y_returner_y"] ** 2
    )
    
    df["close_to_side_line"] = df["ball_bounce_y"].apply(
    lambda y_coord: 1
    if (
        (y_coord >= singles_sideline_y - tolerance)
        | (y_coord <= -singles_sideline_y + tolerance)
    )
    else 0
)
    df["close_to_center_line"] = df["ball_bounce_y"].apply(
        lambda y_coord: 1 if ((y_coord >= -tolerance) & (y_coord <= tolerance)) else 0
    )
    df["close_to_service_line"] = df["ball_bounce_x"].apply(
        lambda x_coord: 1 if (x_coord >= service_line_x - tolerance) else 0
    )
    
    df = binning(
        df=df,
        column="ball_hit_v",
        new_column_name="bin_mean_speed",
        step_size=1,
        round_up=1,
        drop_old_column=True,
    )
    df = binning(
        df=df,
        column="ball_hit_y",
        new_column_name="bin_mean_ball_hit_y",
        step_size=0.4,
        round_up=3,
        drop_old_column=True,
    )
    df = binning(
        df=df,
        column="ball_hit_z",
        new_column_name="bin_mean_ball_hit_z",
        step_size=0.1,
        round_up=3,
        drop_old_column=True,
    )
    
    #----------------------------------------------------------------------------------------
    # Scaling
    #----------------------------------------------------------------------------------------
    
    # Copying column names from 3.Feature_Engineering.ipynb
    robust_columns = ["bin_mean_ball_hit_y", "ball_bounce_y", "hitter_y", "receiver_y"]
    standard_columns = [
        "ball_hit_x",
        "ball_net_v",
        "ball_net_z",
        "ball_net_y",
        "ball_bounce_x",
        "ball_bounce_v",
        "ball_bounce_angle",
        "hitter_x",
        "receiver_x",
        "dist_ball_bounce_x_returner_x",
        "dist_ball_bounce_y_returner_y",
        "dist_ball_bounce_returner_total",
        "bin_mean_speed",
        "bin_mean_ball_hit_z",
    ]
    non_scaled_columns = [
        "surface_clay",
        "surface_hard",
        "serve_side_deuce",
        "hitter_hand_left",
        "receiver_hand_left",
        "serve_number",
        "close_to_side_line",
        "close_to_center_line",
        "close_to_service_line",
    ]
    
    # Transform data
    loaded_scaler = joblib.load('scaler.joblib')
    processed_data = loaded_scaler.transform(df)

    # Convert back to a DataFrame with appropriate column names
    df = pd.DataFrame(processed_data, columns=robust_columns + standard_columns + non_scaled_columns)
    
    #----------------------------------------------------------------------------------------
    # Dropping correlated columns
    #----------------------------------------------------------------------------------------
    
    columns_to_drop = [
    "hitter_y",
    "dist_ball_bounce_returner_total",
    "ball_net_v",
    "ball_net_y",
    "ball_bounce_v",
    "serve_side_deuce",
]
    df = df.drop(columns=columns_to_drop)
    
    return df

#############################################################################################
# DISPLAY CORRELATION MATRIX IN CHUNKS
#############################################################################################    
    
def display_correlation_in_chunks(corr_matrix, N=None):
    """
    Displays the correlation matrix in chunks of N rows.

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix
        N (int, optional): Number of rows to display per heatmap. Defaults to full matrix.
    """
    if N is None:
        N = corr_matrix.shape[0]  # Default to full matrix

    total_rows = corr_matrix.shape[0]  # Number of rows in the matrix

    for i in range(0, total_rows, N):  # Iterate in steps of N
        chunk = corr_matrix.iloc[i : i + N, :]  # Select N rows

        # Create a heatmap for the whole matrix/chunk
        if N is None: # Whole matrix
            figsize=(10, 8)  # Adjust height to match chunk size
            annotation_size = 4
        else: # Chunk
            figsize=(12, 3)
            annotation_size = 6
        # Create heatmap
        plt.figure(figsize=figsize)  # Adjust height to match chunk size
        sns.heatmap(
            chunk,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": annotation_size},
            vmin=-1,
            vmax=1,
        )
        plt.title(
            f"Correlation Matrix Heatmap (Rows {i + 1} to {min(i + N, total_rows)})"
        )
        plt.show()

#############################################################################################
# FUNCTION FOR BINNING AND TAKING MEAN OF BINNED DATA
############################################################################################# 
        
def binning(df, column, new_column_name, step_size, round_up=4, drop_old_column=False):
    """
    Bin data as desired based on training data.
    
    Inputs:
    - df : input dataframe
    - column : column for binning
    - new_column_name : name of column for binned and mean transformed data
    - step_size : for binning
    - round_up : number of decimal places to round up
    - drop_old_column : True/False if old column should be dropped
    
    Outputs:
    - dataframe with new binned column
    """
    # Define bin edges (adjust based on min/max from training dataset)
    data = pd.read_csv('datasets/train_dataset.csv')
    bin_edges = np.arange(data[column].min(), data[column].max() + step_size, step_size)

    # Create bin labels
    df["column_binning"] = pd.cut(
        df[column], bins=bin_edges, right=False
    )  # Right=False ensures left-inclusive bins

    # Compute mean serve speed per bin
    bin_means = df.groupby("column_binning")[column].transform("mean")

    # Add new column and drop some
    df[new_column_name] = bin_means.round(round_up)
    df.drop(columns=["column_binning"], inplace=True)
    if drop_old_column:
        df.drop(columns=column, inplace=True)
        
    return df              
