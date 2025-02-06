#import seaborn as sns
import matplotlib.pyplot as plt

def draw_tennis_court(ax=None):
    """
    Draws a tennis court on a given matplotlib Axes.
    If ax is None, creates a new figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Court dimensions
    #court_length = 23.77  # Full length (x-axis)
    #court_width = 10.97  # Full doubles width (y-axis)

    # Net at x=0, baselines at ±11.885m
    baseline_x = 11.885
    service_line_x = 6.4
    singles_sideline_y = 4.115  # Half of singles width (8.23/2)
    doubles_sideline_y = 5.485  # Half of doubles width (10.97/2)
    center_mark_length = 0.2  # Small center marks on baseline

    # Outer court (doubles court)
    ax.plot(
        [-baseline_x, baseline_x], [-doubles_sideline_y, -doubles_sideline_y], "k", lw=2
    )  # Bottom sideline
    ax.plot(
        [-baseline_x, baseline_x], [doubles_sideline_y, doubles_sideline_y], "k", lw=2
    )  # Top sideline
    ax.plot(
        [-baseline_x, -baseline_x], [-doubles_sideline_y, doubles_sideline_y], "k", lw=2
    )  # Left baseline
    ax.plot(
        [baseline_x, baseline_x], [-doubles_sideline_y, doubles_sideline_y], "k", lw=2
    )  # Right baseline

    # Singles sidelines
    ax.plot(
        [-baseline_x, baseline_x], [-singles_sideline_y, -singles_sideline_y], "k", lw=1
    )  # Bottom singles sideline
    ax.plot(
        [-baseline_x, baseline_x], [singles_sideline_y, singles_sideline_y], "k", lw=1
    )  # Top singles sideline

    # Service lines
    ax.plot(
        [service_line_x, service_line_x],
        [-singles_sideline_y, singles_sideline_y],
        "k",
        lw=1,
    )  # Service line near net
    ax.plot(
        [-service_line_x, -service_line_x],
        [-singles_sideline_y, singles_sideline_y],
        "k",
        lw=1,
    )  # Service line far side

    # Center service line
    ax.plot([-service_line_x, service_line_x], [0, 0], "k", lw=1)

    # Net
    ax.plot([0, 0], [-doubles_sideline_y, doubles_sideline_y], "k", lw=3)

    # Center marks on the baselines
    ax.plot(
        [baseline_x - center_mark_length, baseline_x],
        [0, 0],
        "k",
        lw=2,
    )  # Right baseline center mark
    ax.plot(
        [-baseline_x, -baseline_x + center_mark_length],
        [0, 0],
        "k",
        lw=2,
    )  # Left baseline center mark
    
    return fig, ax
