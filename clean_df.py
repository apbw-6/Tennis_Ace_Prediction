import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def clean_dataframe(df):
    """
    - Drop duplicates.
    - z-coordinate cannot be negative.
    - It is not possible that the server and the receiver are on the same side of the court. These points must be deleted.
    - It is not possible that the server and the ball bounce is on the same side of the court. These points must be deleted.
    - <b>Fliping the court</b> to make things simpler. I.e the server will always be on the left and the returner will always be on the right. This shouldn't make any difference to the game.
    - Server position must be close to the baseline and cannot be way inside the court or way behind.
    - Returner position must be near or behind the baseline and cannot be way inside the court.
     Fastest serve speed recorded is 163.7mph. Serve speed cannot exceed this.
    - Impute missing data with mean and most frequent.

    Args:
        datafarme: to be cleaned

    Returns:
        dataframe: cleaned data
    """
    
    # Tennis court dimensions
    baseline_x = 11.885
    service_line_x = 6.4
    singles_sideline_y = 4.115 
    
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
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df