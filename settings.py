# Constants for Data Processing and Visualization

# Delay time in seconds for processing tasks.
# This constant defines the amount of time to wait before executing
# the next operation, which can be useful for managing the flow of
# data processing or visual updates.
DELAY = 0.5

# Probability of generating outliers in the dataset.
# This constant represents the likelihood (between 0 and 1) of 
# introducing outliers into the data during generation or simulation.
# A value of 0.2 indicates a 20% chance of an outlier being created.
OUTLIERS_GENERATION_PROBABILITY = 0.2

# Flag to enable or disable visualization.
# When set to True, this constant allows for the rendering of 
# visual representations of data, such as plots or graphs.
# Setting it to False will suppress any visualization output.
VISUALIZATION = True