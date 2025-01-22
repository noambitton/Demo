import pandas as pd

# Hardcoded binning options with semantic and utility values
binning_options = {
    "option_1": {"semantic": 0, "utility": 0},
    "option_2": {"semantic": 1, "utility": 0.3},
    "option_3": {"semantic": 0.25, "utility": 0.8},
    "option_4": {"semantic": 1, "utility": 0.1},
    "option_5": {"semantic": 0.5, "utility": 0.5},
}

# Convert binning options to a DataFrame
binning_df = pd.DataFrame([{
    "binning_option": name,
    "semantic": values["semantic"],
    "utility": values["utility"]
} for name, values in binning_options.items()])
