import numpy as np
import matplotlib.pyplot as plt

with open('epsilon90.txt') as f:
    lines = f.readlines()
    x1 = [i * 100 for i in range(len(lines[:101]))]
    y1 = [float(line.split(',')[1]) for line in lines[:101]]

with open('epsilon65.txt') as g:
    lines = g.readlines()
    x2 = [i * 100 for i in range(len(lines[:101]))]
    y2 = [float(line.split(',')[1]) for line in lines[:101]]

with open('epsilon35.txt') as h:
    lines = h.readlines()
    x3 = [i * 100 for i in range(len(lines[:101]))]
    y3 = [float(line.split(',')[1]) for line in lines[:101]]        

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_title("Average Score der letzten 100 Spiele")    
ax1.set_xlabel('Games')
ax1.set_ylabel('Average Score')
ax1.plot(x1,y1, c='r', label=r'$\epsilon$ = 0.9')
#ax1.set_xscale('log')
#ax1.set_yscale('symlog')


ax2 = fig.add_subplot(111) 
ax2.set_xlabel('Games')
ax2.set_ylabel('Average Score')
ax2.plot(x2,y2, c='b', label=r'$\epsilon$ = 0.65')
#ax2.set_yscale('symlog')


ax3 = fig.add_subplot(111)
ax3.set_xlabel('Games')
ax3.set_ylabel('Average Score')
ax3.plot(x3,y3, c='g', label=r'$\epsilon$ = 0.35')
#ax3.set_yscale('symlog')


leg = ax1.legend()

plt.show()