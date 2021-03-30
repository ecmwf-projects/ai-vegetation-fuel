from utils.generate_io_arrays import data_combined, data_to_narray
from utils.data_split_by_region import tropics_dataset, midlats_dataset
from utils.fuelload import fuelload
from utils.data_paths import export_data_paths
import os

data_root_path = input("\nEnter root directory where data is stored: ")

# check file path entered
assert os.path.exists(data_root_path) is not False, "Root Directory doesn't exist."

# import file paths
agb_data, ba_data, fuel_load_data, combined_data = export_data_paths()

# path to data files
agb_data_path = data_root_path + agb_data
burned_area_data_path = data_root_path + ba_data
fuel_load_data_path = data_root_path + fuel_load_data

# creating fuel_load datafile
fuel_load_dataset = fuelload(agb_data_path, burned_area_data_path, fuel_load_data_path)
print("\nFuel Load dataset created. Saved to", fuel_load_data_path, "\n")

print("Merging input features...\n")
combined_dataset_path = data_root_path + combined_data
combined_dataset = data_combined(data_root_path, fuel_load_data_path)

# splitting dataset into train, val and test dataframes
df_train, df_val, df_test = data_to_narray(combined_dataset)

print("Processing data for the Tropics region...\n")
# handling tropics
tropic_train, tropic_val, tropic_test = tropics_dataset(
    df_train, df_val, df_test, data_root_path
)
print(
    "Training, Validation and Testing datasets for the Tropics region dataset created, and saved to:",
    data_root_path + "/tropics.",
)
print(
    "Monthly Inference datasets for the Tropics region created, and saved to:",
    data_root_path + "/infer_tropics. \n",
)

print("Processing data for the Mid-Latitudes region...\n")
# handling midlats
midlat_train, midlat_val, midlat_test = midlats_dataset(
    df_train, df_val, df_test, data_root_path
)
print(
    "Training, Validation and Testing datasets for the Mid-Latitudes region dataset created, and saved to:",
    data_root_path + "/midlats.",
)
print(
    "Monthly Inference datasets for the Mid-Latitudes region created, and saved to:",
    data_root_path + "/infer_midlats.\n",
)
