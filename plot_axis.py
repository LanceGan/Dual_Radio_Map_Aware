import matplotlib.pyplot as plt
import numpy as np

result = "data"
x = [100, 200, 300]
if result == 'data':
    A = [794.36/174.32,1368.23/215.42,1957.82/259.94]
    B = [732.47/192.25,1241.40/228.16,1872.40/288.22]
    C = [889.24/209.98,1405.23/263.96,2002.70/320.06]
    
    # Ours = [174.32, 215.42, 259.94]
    # PSO = [192.25, 228.16, 288.22]
    # ACO = [209.98, 263.96, 320.06] 
else :
    A = [26.73,38.47,60.55]
    B = [33.94,50.28,73.68]
    C = [53.94,69.5,78.71]

plt.figure(figsize=(7,5))

if result == 'data':
    plt.plot(x, A, marker='o', label='Proposed Algorithm', color='#e41a1c', linewidth=2.5, markersize=8,linestyle='--')
    plt.plot(x, B, marker='s', label='DRL-SA', color='#dede00', linewidth=2.5, markersize=8,linestyle='--')
    plt.plot(x, C, marker='^', label='TSP', color='#4daf4a', linewidth=2.5, markersize=8,linestyle='--')

    for i, value in enumerate(A):
        plt.annotate(f'{value:.2f}', (x[i], A[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    for i, value in enumerate(B):
        plt.annotate(f'{value:.2f}', (x[i], B[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    for i, value in enumerate(C):
        plt.annotate(f'{value:.2f}', (x[i], C[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    
    # plt.xlabel('Data Size (MB)', fontsize=13)
    plt.ylabel('Average Data Rate (MB/s)', fontsize=13)
    plt.xticks(x, ['I=100MB', 'I=200MB', 'I=300MB'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='best', frameon=True, facecolor='whitesmoke', edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figs/'+result+'.pdf',dpi=300, bbox_inches='tight')
else :
    plt.plot(x, A, marker='o', label='Proposed Algorithm', color='#e41a1c', linewidth=2.5, markersize=8,linestyle='--')
    plt.plot(x, B, marker='s', label='DRL-SA', color='#dede00', linewidth=2.5, markersize=8,linestyle='--')
    plt.plot(x, C, marker='^', label='TSP', color='#4daf4a', linewidth=2.5, markersize=8,linestyle='--')

    for i, value in enumerate(A):
        plt.annotate(f'{value:.2f}', (x[i], A[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    for i, value in enumerate(B):
        plt.annotate(f'{value:.2f}', (x[i], B[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    for i, value in enumerate(C):
        plt.annotate(f'{value:.2f}', (x[i], C[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10)
    
    
    # plt.xlabel('Data Size (MB)', fontsize=13)
    plt.ylabel('Outage Time (s)', fontsize=13)
    plt.xticks(x, ['I=100MB', 'I=200MB', 'I=300MB'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=80, color='black', linestyle='--', linewidth=1.5, label='Tolerance Limit')
    plt.legend(fontsize=11, loc='best', frameon=True, facecolor='whitesmoke', edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figs/'+result+'.pdf',dpi=300, bbox_inches='tight')
