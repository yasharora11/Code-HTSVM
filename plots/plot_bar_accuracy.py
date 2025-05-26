import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file (generate this file by running the code under analysis section)
df = pd.read_excel('data/HTSVM_varying_delta_lambda.xlsx')

# Extract the delta values and accuracies
delta_values = df['Delta'].to_numpy()
hsvm_acc = df['HSVM_Breast cancer'].to_numpy()
htsvm_acc = df['HTSVM_Breast cancer'].to_numpy()

# Convert delta to log scale labels like 10^{-4}
x_labels = [f'$10^{{{int(np.log10(d))}}}$' for d in delta_values]
x_pos = np.arange(len(delta_values))

# Plot settings
bar_width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x_pos - bar_width/2, hsvm_acc, width=bar_width, label='HSVM', color='red')
plt.bar(x_pos + bar_width/2, htsvm_acc, width=bar_width, label='HTSVM', color='blue')

plt.xticks(x_pos, x_labels)
plt.ylabel('Average Accuracy')
plt.xlabel(r'$\delta$')
plt.title('(a) Breast cancer')
plt.legend()

plt.tight_layout()
plt.show()
