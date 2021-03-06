import numpy as np
import matplotlib.pyplot as plt

with open('epsilon90.txt') as f:
    lines = f.readlines()
    x1 = [i * 100 for i in range(len(lines[:101]))]
    total = 0
    y1 = []
    for line in lines[:101]:
        total += float(line.split(',')[0])
        y1.append(total)

with open('epsilon60.txt') as g:
    lines = g.readlines()
    x2 = [i * 100 for i in range(len(lines[:101]))]
    total = 0
    y2 = []
    for line in lines[:101]:
        total += float(line.split(',')[0])
        y2.append(total)

with open('epsilon35.txt') as h:
    lines = h.readlines()
    x3 = [i * 100 for i in range(len(lines[:101]))]
    total = 0
    y3 = []
    for line in lines[:101]:
        total += float(line.split(',')[0])
        y3.append(total)


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_title("winrate with different epsilons")    
ax1.set_xlabel('games')
ax1.set_ylabel('wins per 100 games')
ax1.plot(x1,y1, c='r', label='epsilon = 0.9')
#ax1.set_xscale('log')
#ax1.set_yscale('log')


ax2 = fig.add_subplot(111) 
ax2.set_xlabel('games')
ax2.set_ylabel('wins per 100 games')
ax2.plot(x2,y2, c='b', label='epsilon = 0.65')
#ax2.set_xscale('log')
#ax2.set_yscale('log')


ax3 = fig.add_subplot(111)
ax3.set_xlabel('games')
ax3.set_ylabel('wins per 100 games')
ax3.plot(x3,y3, c='g', label='epsilon = 0.35')
#ax3.set_xscale('log')
#ax3.set_yscale('log')


leg = ax1.legend()

plt.show()