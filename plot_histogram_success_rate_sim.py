import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(0)

# Generate dummy success rate data for each method
frogger_data = np.random.normal(loc=0.6, scale=0.1, size=250)
dex_diffuser_data = np.random.normal(loc=0.65, scale=0.15, size=250)
bps_pgs_evaluator_data = np.random.normal(loc=0.7, scale=0.1, size=250)
gg_data = np.random.normal(loc=0.85, scale=0.05, size=250)  # Best performance for GG-Ours

# Define bins for histogram
bins = np.linspace(0, 1, 21)  # Bin width of approximately 0.05

# Create the histogram plot
plt.figure(figsize=(14, 3))
plt.hist(frogger_data, bins, alpha=0.7, label='Frogger (Analytic/Mesh)', color='royalblue')
plt.hist(dex_diffuser_data, bins, alpha=0.7, label='DexDiffuser (Generative/BPS)', color='darkgreen')
plt.hist(bps_pgs_evaluator_data, bins, alpha=0.7, label='GG-Ablation (PGS/BPS)', color='goldenrod')
plt.hist(gg_data, bins, alpha=0.7, label='GG-Ours (PGS/NeRF)', color='firebrick')

# Adding titles and labels
plt.xlabel('Per-object Success Rate')
plt.ylabel('Frequency')
plt.title('Per-Object Success Rate in Simulation')
plt.legend()

# Improve aesthetics
plt.grid(True)
plt.style.use('seaborn-v0_8-darkgrid')
plt.tight_layout()

# Save the plot to a PDF with high resolution
file_path = "Per_Object_Success_Rate_in_Simulation.pdf"
plt.savefig(file_path, format='pdf', dpi=300)

plt.show()
