# Import everything from my_libraries.py
from my_libraries import *

# Constants
baseline_x = 11.885
service_line_x = 6.4
singles_sideline_y = 4.115
tolerance = 0.25  # Setting tolerance of 25cm (for new features)

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
     Fastest serve speed recorded is 163.7mph. Serve speed cannot exceed this.

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
    
    # It is not possible for ball speed to be greater at the net than when the server has just hit the ball. These points must be deleted.
    df = df[df["ball_hit_v"] > df["ball_net_v"]]    
    
    return df

def impute_data(df):
    """ - Impute missing data with mean and most frequent. """
    
    # Load the saved imputer
    loaded_imputer = joblib.load("simple_imputer.joblib")

    cols_most_frequent = [
        "surface",
        "serve_side",
        "serve_number",
        "hitter_hand",
        "receiver_hand",
    ]
    df[cols_most_frequent] = loaded_imputer.transform(df[cols_most_frequent])
    
    # Impute missing values with column mean (only for numeric columns)
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
    
    #----------------------------------------------------------------------------------------
    # Scaling
    #----------------------------------------------------------------------------------------
    
    # Copying column names from 3.Feature_Engineering.ipynb
    robust_columns = ["ball_hit_y", "ball_bounce_y", "hitter_y", "receiver_y"]
    standard_columns = [
        "ball_hit_x",
        "ball_hit_z",
        "ball_hit_v",
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
        "ball_bounce_v"
    ]
    df = df.drop(columns=columns_to_drop)
    
    return df

#############################################################################################
# MODEL SCORES
#############################################################################################

def scoring():
    """ A function to print scores and save them in lists. """
    
    print("-----Test Data Accuracy----")
    a_s = round(accuracy_score(y_test.to_numpy(), y_pred), 4)
    print('Accuracy score:', a_s)
    f_1 = round(f1_score(y_test.to_numpy(), y_pred), 4)
    print('F1 score:', f_1)
    accuracy_test.append(a_s)
    F1score_test.append(f_1)

    print("\n-----Train Data Accuracy----")
    a_s = round(accuracy_score(y_train.to_numpy(), y_pred_train), 4)
    print('Accuracy score:', a_s)
    f_1 = round(f1_score(y_train.to_numpy(), y_pred_train), 4)
    print('F1 score:', f_1)
    accuracy_train.append(a_s)
    F1score_train.append(f_1)