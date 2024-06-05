import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

# Success data from the table
easy_data = {
    'Frogger': [5, 5, 3, 4, 4, 3],
    'DexDiffuser': [4, 5, 4, 5, 3, 1],
    'GG-Ours': [5, 5, 5, 4, 5, 4]
}

medium_data = {
    'Frogger': [1, 4, 2, 5, 2, 5],
    'DexDiffuser': [5, 0, 4, 3, 0, 1],
    'GG-Ours': [5, 4, 5, 5, 2, 5]
}

hard_data = {
    'Frogger': [0, 0, 0, 0, 1, 0, 5, 2],
    'DexDiffuser': [1, 3, 0, 0, 2, 0, 0, 2],
    'GG-Ours': [4, 5, 0, 2, 3, 2, 1, 5]
}

# Calculate success rates
def calculate_success_rate(data):
    total_success = sum(data)
    total_attempts = len(data) * 5
    return total_success / total_attempts

success_rates = {
    'Frogger (Analytic/Mesh)': [
        calculate_success_rate(easy_data['Frogger']),
        calculate_success_rate(medium_data['Frogger']),
        calculate_success_rate(hard_data['Frogger'])
    ],
    'DexDiffuser (Generative/BPS)': [
        calculate_success_rate(easy_data['DexDiffuser']),
        calculate_success_rate(medium_data['DexDiffuser']),
        calculate_success_rate(hard_data['DexDiffuser'])
    ],
    'Get a Grip (Ours)': [
        calculate_success_rate(easy_data['GG-Ours']),
        calculate_success_rate(medium_data['GG-Ours']),
        calculate_success_rate(hard_data['GG-Ours'])
    ]
}

# Data for plotting
object_difficulties = ['Easy', 'Medium', 'Hard']
x = np.arange(len(object_difficulties))  # the label locations
width = 0.2  # the width of the bars

# Apply seaborn style
sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(24, 4))  # Adjusted size for shorter and wider aspect ratio

# Plotting each method
rects1 = ax.bar(x - width, success_rates['Frogger (Analytic/Mesh)'], width, label='Frogger (Analytic/Mesh)')
rects2 = ax.bar(x, success_rates['DexDiffuser (Generative/BPS)'], width, label='DexDiffuser (Generative/BPS)')
rects3 = ax.bar(x + width, success_rates['Get a Grip (Ours)'], width, label='Get a Grip (Ours)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Object Difficulties', fontsize=24)
ax.set_ylabel('Success Rate', fontsize=24)
ax.set_title('Success Rate on Hardware', fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(object_difficulties, fontsize=20)
ax.set_yticklabels(np.arange(0, 1.1, step=0.1), fontsize=20)
ax.set_ylim(0, 1)
ax.legend(fontsize=20)

# Adding values on top of the bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

def format_func(value, tick_number):
    return f"{value:.1f}"

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(format_func)
)  # 1 decimal place

# Save the adjusted plot to a PDF file with 300 dpi and tight layout
fig.savefig('Success_Rate_on_Hardware_Three_Methods.pdf', dpi=300, bbox_inches='tight')
plt.show()

