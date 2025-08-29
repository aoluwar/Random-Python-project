import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_ambient_temperature(days, start_date='2019-01-01'):
    """Generate realistic ambient temperature data based on seasonal patterns"""
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    day_of_year = dates.dayofyear
    
    # Seasonal temperature pattern (sinusoidal with noise)
    base_temp = 7.5  # Average temperature from paper
    seasonal_amplitude = 15  # Temperature variation
    seasonal_temp = base_temp + seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Add daily variation and noise
    daily_noise = np.random.normal(0, 3, days)
    ambient_temp = seasonal_temp + daily_noise
    
    # Clamp to realistic range
    ambient_temp = np.clip(ambient_temp, -25, 35)
    
    return dates, day_of_year, ambient_temp

def calculate_layer_temperature(ambient_temp, day_of_year, depth, material_type='control', density=None):
    """Calculate layer temperature based on ambient conditions and material properties"""
    
    # Thermal lag effect based on depth
    thermal_lag_days = depth * 2  # Deeper layers have more lag
    lag_indices = np.maximum(0, np.arange(len(ambient_temp)) - int(thermal_lag_days))
    lagged_ambient = ambient_temp[lag_indices]
    
    # Damping factor based on depth and material
    if material_type == 'control':
        damping_factor = 0.85 * np.exp(-depth * 1.2)
        insulation_effect = 0
    else:  # LCC material
        # Better insulation properties for LCC
        base_damping = 0.9 * np.exp(-depth * 0.8)
        if density == 400:
            insulation_effect = 0.15
        elif density == 475:
            insulation_effect = 0.12
        elif density == 600:
            insulation_effect = 0.08
        else:
            insulation_effect = 0.12
        damping_factor = base_damping * (1 + insulation_effect)
    
    # Seasonal phase shift based on depth
    phase_shift = depth * 10  # days
    seasonal_component = 2 * np.sin(2 * np.pi * (day_of_year - 80 - phase_shift) / 365)
    
    # Calculate layer temperature
    layer_temp = (lagged_ambient * damping_factor + 
                  seasonal_component * (1 - damping_factor) + 
                  np.random.normal(0, 0.5, len(ambient_temp)))
    
    return layer_temp

def generate_control_section_data(days=1000):
    """Generate temperature data for control section (conventional pavement)"""
    dates, day_of_year, ambient_temp = generate_ambient_temperature(days)
    
    # Layer depths from the paper (in meters)
    depths = {
        'base_0.225': 0.225,
        'subbase_0.7': 0.7,
        'subgrade_0.85': 0.85
    }
    
    data = {
        'date': dates,
        'day_of_year': day_of_year,
        'ambient_temp': ambient_temp
    }
    
    # Generate temperature for each layer
    for layer_name, depth in depths.items():
        layer_temp = calculate_layer_temperature(ambient_temp, day_of_year, depth, 'control')
        data[f'temp_{layer_name}'] = layer_temp
        data[f'depth_{layer_name}'] = [depth] * days
    
    return pd.DataFrame(data)

def generate_lcc_section_data(days=1000, density=475):
    """Generate temperature data for LCC section"""
    dates, day_of_year, ambient_temp = generate_ambient_temperature(days)
    
    # Layer depths for LCC section (in meters)
    depths = {
        'asphalt_0.075': 0.075,
        'base_0.225': 0.225,
        'subbase_0.475': 0.475,
        'subgrade_0.75': 0.75
    }
    
    data = {
        'date': dates,
        'day_of_year': day_of_year,
        'ambient_temp': ambient_temp,
        'lcc_density': [density] * days
    }
    
    # Generate temperature for each layer
    for layer_name, depth in depths.items():
        layer_temp = calculate_layer_temperature(ambient_temp, day_of_year, depth, 'lcc', density)
        data[f'temp_{layer_name}'] = layer_temp
        data[f'depth_{layer_name}'] = [depth] * days
    
    return pd.DataFrame(data)

def generate_ndd_validation_data(days=365):
    """Generate validation data for Notre Dame Drive (NDD) sections"""
    datasets = {}
    
    # Control section
    datasets['ndd_control'] = generate_control_section_data(days)
    
    # LCC sections with different densities
    for density in [400, 475, 600]:
        datasets[f'ndd_lcc_{density}'] = generate_lcc_section_data(days, density)
    
    return datasets

# Generate main datasets
print("Generating Erbsville Control Section Dataset...")
control_data = generate_control_section_data(1000)

print("Generating Erbsville LCC Section Dataset...")
lcc_data = generate_lcc_section_data(1000, density=475)

print("Generating Notre Dame Drive Validation Datasets...")
ndd_datasets = generate_ndd_validation_data(365)

# Display sample data and statistics
print("\n=== CONTROL SECTION SAMPLE DATA ===")
print(control_data.head())
print(f"\nControl Data Shape: {control_data.shape}")

print("\n=== LCC SECTION SAMPLE DATA ===")
print(lcc_data.head())
print(f"\nLCC Data Shape: {lcc_data.shape}")

print("\n=== DATA STATISTICS SUMMARY ===")
print("\nControl Section Temperature Statistics:")
temp_cols = [col for col in control_data.columns if col.startswith('temp_')]
print(control_data[temp_cols + ['ambient_temp']].describe())

print("\nLCC Section Temperature Statistics:")
temp_cols_lcc = [col for col in lcc_data.columns if col.startswith('temp_')]
print(lcc_data[temp_cols_lcc + ['ambient_temp']].describe())

# Save datasets to CSV
control_data.to_csv('erbsville_control_temperature_data.csv', index=False)
lcc_data.to_csv('erbsville_lcc_temperature_data.csv', index=False)

for name, dataset in ndd_datasets.items():
    dataset.to_csv(f'{name}_temperature_data.csv', index=False)

print("\n=== DATASETS GENERATED SUCCESSFULLY ===")
print("Files created:")
print("- erbsville_control_temperature_data.csv")
print("- erbsville_lcc_temperature_data.csv")
print("- ndd_control_temperature_data.csv")
print("- ndd_lcc_400_temperature_data.csv")
print("- ndd_lcc_475_temperature_data.csv")
print("- ndd_lcc_600_temperature_data.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Control section temperature profile
ax1 = axes[0, 0]
ax1.plot(control_data['day_of_year'], control_data['ambient_temp'], label='Ambient', linewidth=1)
for col in temp_cols:
    layer_name = col.replace('temp_', '').replace('_', ' ')
    ax1.plot(control_data['day_of_year'], control_data[col], label=layer_name, alpha=0.8)
ax1.set_title('Control Section Temperature Profiles')
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# LCC section temperature profile
ax2 = axes[0, 1]
ax2.plot(lcc_data['day_of_year'], lcc_data['ambient_temp'], label='Ambient', linewidth=1)
for col in temp_cols_lcc:
    layer_name = col.replace('temp_', '').replace('_', ' ')
    ax2.plot(lcc_data['day_of_year'], lcc_data[col], label=layer_name, alpha=0.8)
ax2.set_title('LCC Section Temperature Profiles')
ax2.set_xlabel('Day of Year')
ax2.set_ylabel('Temperature (°C)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Temperature distribution comparison
ax3 = axes[1, 0]
control_temp_data = control_data[temp_cols].values.flatten()
lcc_temp_data = lcc_data[temp_cols_lcc].values.flatten()
ax3.hist(control_temp_data, bins=30, alpha=0.6, label='Control', density=True)
ax3.hist(lcc_temp_data, bins=30, alpha=0.6, label='LCC', density=True)
ax3.set_title('Temperature Distribution Comparison')
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Correlation matrix for LCC data
ax4 = axes[1, 1]
corr_cols = ['ambient_temp'] + temp_cols_lcc
corr_matrix = lcc_data[corr_cols].corr()
im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_cols)))
ax4.set_yticks(range(len(corr_cols)))
ax4.set_xticklabels([col.replace('temp_', '').replace('_', '\n') for col in corr_cols], rotation=45)
ax4.set_yticklabels([col.replace('temp_', '').replace('_', ' ') for col in corr_cols])
ax4.set_title('Temperature Correlation Matrix (LCC)')
plt.colorbar(im, ax=ax4)

# Add correlation values
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', 
                color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('pavement_temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'pavement_temperature_analysis.png'")
print("\n=== DATASET GENERATION COMPLETE ===")
