import pandas as pd
import numpy as np
import sys

## Test script - creates .csv file to parse into spotpy to run manual sensitivity analysis# Define parameter ranges
param_ranges = {
    "precip_scaling": (0.5, 2.0),  # Updated linear range
    "albedo_ice": (0.1, 0.46),
    "albedo_fresh_snow": (0.75, 0.98),
    "albedo_firn": (0.46, 0.75),
    "albedo_aging": (1, 25),
    "albedo_depth": (1, 15),
    "z0_snow": (0.02, 1.6),
    "z0_ice": (0.7, 20),
    "z0_firn": (1.6, 6.5),
    "aging_z0": (0.0013, 0.0039)
}

# Define precipitation scaling reference values
precip_reference_values = [0.5, 1.25, 2.0]  # Example reference values

# Compute baseline (median) values
baseline = {key: (v[0] + v[1]) / 2 for key, v in param_ranges.items()}

# Generate parameter variations
num_steps = 10
sensitivity_results = []
param_combinations = []

for precip_scaling in precip_reference_values:
    baseline["precip_scaling"] = precip_scaling  # Set precipitation scaling reference value
    
    for param, (p_min, p_max) in param_ranges.items():
        p_mid = baseline[param]
        param_values = np.linspace(p_min, p_max, num_steps)  # Use linear spacing for all parameters
        
        # Ensure baseline is included
        if p_mid not in param_values:
            param_values = np.sort(np.append(param_values, p_mid))
        
        for val in param_values:
            perturbed_params = baseline.copy()
            perturbed_params[param] = val
            
            sensitivity_results.append({
                "precip_scaling_ref": precip_scaling,
                "parameter": param,
                "value": val
            })
            param_combinations.append(perturbed_params)

# Convert results to DataFrame
results_df = pd.DataFrame(sensitivity_results)
combinations_df = pd.DataFrame(param_combinations)

# Rename and reorder columns
column_mapping = {
    "precip_scaling": "parRRR_factor",
    "albedo_ice": "paralb_ice",
    "albedo_fresh_snow": "paralb_snow",
    "albedo_firn": "paralb_firn",
    "albedo_aging": "paralbedo_aging",
    "albedo_depth": "paralbedo_depth",
    "z0_snow": "parroughness_fresh_snow",
    "z0_ice": "parroughness_ice",
    "z0_firn": "parroughness_firn",
    "aging_z0": "paraging_factor_roughness"
}
combinations_df = combinations_df.rename(columns=column_mapping)
combinations_df = combinations_df[list(column_mapping.values())]

# Save results
#sensitivity_df.to_csv("sensitivity_results.csv", index=False)
combinations_df = combinations_df.drop_duplicates(keep="first")

if 'win' in sys.platform:
    combinations_df.to_csv("E:/OneDrive/PhD/PhD/Data/Hintereisferner/COSIPY/MiscTests/manual_sens_params_fullprior.csv", index=False)
else:
    combinations_df.to_csv("/mnt/C4AEBBABAEBB9500/OneDrive/PhD/PhD/Data/Hintereisferner/COSIPY/MiscTests/manual_sens_params_fullprior.csv", index=False)

print("Sensitivity analysis complete. Results saved to 'sensitivity_results.csv' and 'parameter_combinations.csv'")