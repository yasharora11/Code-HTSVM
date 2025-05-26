import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
DATA_PATH = 'data/HTSVM_varying_delta_lambda.xlsx'
df = pd.read_excel(DATA_PATH)

# Prepare axes
x = np.log10(df['Delta'])
y = df['lambda']
z_htsvm = df['HTSVM_Hepatitis']
z_hsvm = df['HSVM_Hepatitis']

# Create figure with two subplots side-by-side
fig = plt.figure(figsize=(10, 4.5))

# Plot HTSVM
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_trisurf(x, y, z_htsvm, cmap=plt.cm.viridis, linewidth=0.2)
ax1.set_xlabel('log10(Delta)')
ax1.set_ylabel('Lambda')
ax1.set_zlabel('Accuracy (%)')
ax1.set_title('HTSVM')
fig.colorbar(surf1, ax=ax1, shrink=0.6)

# Plot HSVM
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_trisurf(x, y, z_hsvm, cmap=plt.cm.viridis, linewidth=0.2)
ax2.set_xlabel('log10(Delta)')
ax2.set_ylabel('Lambda')
ax2.set_zlabel('Accuracy (%)')
ax2.set_title('HSVM')
fig.colorbar(surf2, ax=ax2, shrink=0.6)

# Overall title and layout
plt.suptitle('(c) Hepatitis', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure (optional)
# plt.savefig('HTSVM_HSVM_Horse_3D.png', dpi=300, bbox_inches='tight')

plt.show()
