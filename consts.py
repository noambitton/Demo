import pandas as pd

# Hardcoded binning options with semantic and utility values
binning_options = [
    {"ID": 0, "Semantic": 1.0, "Utility": 0.1698431361, "Pareto": 1,
     "Partition": [-1.0, 18.0, 35.0, 50.0, 65.0, 100.0]},
    {"ID": 2, "Semantic": 0.4431820943, "Utility": 0.8859527461, "Pareto": 1, "Partition": [20.0, 29.0, 81.0]},
    {"ID": 5, "Semantic": 0.4431820943, "Utility": 0.8859527461, "Pareto": 1, "Partition": [20.0, 29.0, 81.0]},
    {"ID": 6, "Semantic": 0.442003157, "Utility": 1.0, "Pareto": 1, "Partition": [20.0, 28.5, 82.0]},
    {"ID": 10, "Semantic": 0.4994929147, "Utility": 0.4039718156, "Pareto": 1, "Partition": [20.0, 25.0, 36.0, 81.0]},
    {"ID": 11, "Semantic": 0.4983130358, "Utility": 0.5346785555, "Pareto": 1, "Partition": [20.0, 24.0, 30.0, 82.0]},
    {"ID": 13, "Semantic": 0.4994929147, "Utility": 0.4039718156, "Pareto": 1, "Partition": [20.0, 25.0, 36.0, 81.0]},
    {"ID": 14, "Semantic": 0.485644939, "Utility": 0.544618926, "Pareto": 1, "Partition": [20.0, 28.5, 62.5, 82.0]},
    {"ID": 16, "Semantic": 0.5850628193, "Utility": 0.3461278883, "Pareto": 1,
     "Partition": [20.0, 21.5, 30.5, 53.5, 82.0]},
    {"ID": 19, "Semantic": 0.5844423071, "Utility": 0.3852705203, "Pareto": 1,
     "Partition": [20.0, 24.0, 30.0, 54.0, 82.0]},
    {"ID": 27, "Semantic": 0.872897431, "Utility": 0.3043894221, "Pareto": 1,
     "Partition": [20.0, 24.0, 30.0, 42.0, 54.0, 82.0]},
    {"ID": 30, "Semantic": 0.8792808497, "Utility": 0.2959928833, "Pareto": 1,
     "Partition": [20.0, 24.5, 28.5, 42.5, 62.5, 82.0]}
]

binning_df = pd.DataFrame(binning_options)
#
# # Hardcoded binning options with semantic and utility values
# binning_options = {
#     "1": {"semantic": 0, "utility": 0},
#     "2": {"semantic": 1, "utility": 0.3},
#     "3": {"semantic": 0.25, "utility": 0.8},
#     "4": {"semantic": 1, "utility": 0.1},
#     "5": {"semantic": 0.5, "utility": 0.5},
# }
#
# # Convert binning options to a DataFrame
# binning_df = pd.DataFrame([{
#     "binning_option": name,
#     "semantic": values["semantic"],
#     "utility": values["utility"]
# } for name, values in binning_options.items()])
