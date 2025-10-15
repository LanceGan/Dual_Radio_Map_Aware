import matplotlib.pyplot as plt
import numpy as np


result = 'delay'
labels = ['I=100MB', 'I=200MB', 'I=300MB']

if result == 'delay':
    #传输时延
    Ours = [92.23, 134.81, 157.63]
    PSO = [107.57, 144.83, 164.55]
    ACO = [124.24, 164.3, 182.09]
else:
   #飞行时间
    Ours = [174.32, 215.42, 259.94]
    PSO = [192.25, 228.16, 288.22]
    ACO = [209.98, 263.96, 320.06] 

x = np.arange(len(labels))
width = 0.25

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(7,5))

if result == 'delay':
    rects1 = ax.bar(x - width, Ours, width, label='Proposed Algorithm', color="#1f77b4", edgecolor='black')
    rects2 = ax.bar(x, PSO, width, label='DRL-SA', color='#ff7f0e', edgecolor='black')
    rects3 = ax.bar(x + width, ACO, width, label='TSP', color='#7f7f7f', edgecolor='black')
    ax.bar_label(rects1, fmt='%.2f', fontsize=10)
    ax.bar_label(rects2, fmt='%.2f', fontsize=10)
    ax.bar_label(rects3, fmt='%.2f', fontsize=10)
    ax.set_ylabel('Transmission Delay (s)', fontsize=14)
else:
    rects1 = ax.bar(x - width, Ours, width, label='Proposed Algorithm', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x, PSO, width, label='DRL-SA', color='#ff7f0e', edgecolor='black')
    rects3 = ax.bar(x + width, ACO, width, label='TSP', color='#7f7f7f', edgecolor='black')
    ax.bar_label(rects1, fmt='%.2f', fontsize=10)
    ax.bar_label(rects2, fmt='%.2f', fontsize=10)
    ax.bar_label(rects3, fmt='%.2f', fontsize=10)
    ax.set_ylabel('Task Completion Time (s)', fontsize=14)


# ax.set_xlabel('Method', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.legend(fontsize=12)
# ax.set_title('Comparison of Transmission Delay', fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Add value labels
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)
# plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
if result == 'delay':
    plt.savefig('results/figs/delay_comparison.pdf', dpi=300, bbox_inches='tight')
else:  
    plt.savefig('results/figs/flight_time_comparison.pdf', dpi=300, bbox_inches='tight')

plt.show()