import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

#Python script for plotting the performance.

def save(name):
    fig.savefig('{}.png'.format(name), format='png', dpi=1200, bbox_inches='tight')


input_filename = sys.argv[1]
perf_filenames = sys.argv[2:]


fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Elements/Process')
ax.set_ylabel('[GFlops/s]', rotation=0, ha='left')
ax.set_title('Performance', loc='left', fontdict={'weight':'bold'}, pad=25)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_coords(0, 1.02)

ax.set_facecolor((0.9, 0.9, 0.9))
#ax.set_yticks(np.arange(0, 11, 1))
plt.grid(color='w', axis='y')
plt.ylim([0, 68])

input_sizes = np.loadtxt(input_filename);
results = [np.loadtxt(f) for f in perf_filenames]
labels = ['Single 7load1write','Double Ax_mem', 'Optimized performance 274MHz']
# log-scale x
ax.set_xscale('log', basex=2)
#plt.xticks(input_sizes)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.margins(x=0.04)
for i in range(len(results)):
    ax.plot(input_sizes, results[i]/1000, '-o',label=labels[i])


plt.legend()
plt.show()
save('perfnx8')
