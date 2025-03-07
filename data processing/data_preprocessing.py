# Prep datasets for training and testing
# Create train, test, validation splits of the data for specified groupings of sensors


# Group 1 (All sensors in cooling and fueling systems): temp delta, cooling flow rate, cooling pump current and pressure delta, fuel lp flow, fuel hp rail, fuel hp relief, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 2 (Current and pressure sensors only): cooling pump current, cooling pressure delta, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 3 (current sensors only): cooling pump current, fuel lp current, fuel hp current


# Reads in the sensor values for each opeartional profile from the specified directory and their associated sequence failures
# Returns dict of {operational profile: list of failure profiles and sequence point failures}
def parse_avg_sensor_data(dir, sequence_failures):
    pass


# Create train, test, validation splits of the data for specified groupings of sensors
def create_train_test_val_splits(data, sensor_group):
    pass