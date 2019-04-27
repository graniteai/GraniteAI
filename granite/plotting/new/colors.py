# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:57:16 2019

@author: Mike Staddon
"""

from seaborn import cubehelix_palette

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set style
style = {'axes.axisbelow': True,
         'axes.edgecolor': '.85',
         'axes.facecolor': 'white',
         'axes.grid': True,
         'axes.spines.bottom': True,
         'axes.spines.left': True,
         'axes.spines.right': True,
         'axes.spines.top': True,
         'figure.facecolor': 'white',
         'font.size': 12,
         'grid.color': '.85',
         'grid.linestyle': '-',
         'text.color': '.15',
         'xtick.bottom': False,
         'xtick.top': False,
         'ytick.left': False,
         'ytick.right': False}

for s in style:
    matplotlib.rcParams[s] = style[s]


# HSL colors
red = (350, 60, 45)
orange = (30, 90, 50)
yellow = (50, 85, 50)
green = (110, 50, 35)
cyan = (190, 50, 70)
blue = (210, 80, 30)
pink = (310, 70, 80)
purple = (270, 55, 55)
brown = (10, 30, 40)
grey = (0, 0, 65)

def hsl_to_rgb(hsl):
    h, s, l = hsl
    
    h = h % 360
    s = s / 100
    l = l / 100
    
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = l - c / 2
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
        
    return (r + m, g + m, b + m)

hsl_colors = [blue,
              orange,
              red,
              green,
              yellow,
              purple,
              pink,
              brown,
              cyan,
              grey]

colors = [hsl_to_rgb(col) for col in hsl_colors]

categoric_cmap = LinearSegmentedColormap.from_list('default', colors, N=len(colors))

numeric_cmap = cubehelix_palette(start=2.5, rot=2/3,
                                 light=0.25, dark=0.75, hue=1.5,
                                     as_cmap=True)


categoric_palette = colors


if __name__ == '__main__':
    import numpy as np
    
    # Colours in lab space
    hsl = np.array(hsl_colors)
    theta, r = hsl[:, 0] * np.pi / 180, hsl[:, 1]
    plt.scatter(r * np.cos(theta), r * np.sin(theta), c=colors, s=100)
    plt.xlim(-125, 125)
    plt.ylim(-125, 125)
    plt.xticks([-100, 0, 100])
    plt.yticks([-100, 0, 100])
    plt.gcf().set_size_inches(5, 5)
    
    theta = np.linspace(0, 2*np.pi)
    plt.plot(100*np.cos(theta), 100*np.sin(theta), color='k')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show()
    
    # Brightness
    plt.scatter([i for i in range(len(colors))], hsl[:, 2], c=colors)
    plt.ylim(0, 100)
    plt.ylabel('L')
    plt.show()
    
    
    
    fig, axes = plt.subplots(3, 3)
    
    # Fake histogram
    bottom = 0
    for i, color in enumerate(colors):
        x = np.random.normal(loc=i, scale=5, size=10000)
        counts, edges = np.histogram(x, bins=25, range=(-10, 9 + 10))
        axes[0, 0].bar(edges[:-1], counts, width=edges[1] - edges[0], color=color, bottom=bottom)
        
        bottom += counts
    
    for i, color in enumerate(colors):
        x = np.random.normal(size=10) + i
        y = np.random.normal(size=10)
        axes[0, 1].scatter(x, y, c=color)

    for i, color in enumerate(colors):
        x = np.linspace(0, 1)
        y = x + np.random.normal(scale=0.1, size=50) + i
        
        axes[0, 2].plot(x, y, color=color, lw=3)

    # And the numeric cmap
    bottom = 0
    for i in range(10):
        x = np.random.normal(loc=i, scale=5, size=10000)
        counts, edges = np.histogram(x, bins=25, range=(-10, 9 + 10))
        axes[1, 0].bar(edges[:-1], counts, width=edges[1] - edges[0],
            color=numeric_cmap(i/9), bottom=bottom)
        
        bottom += counts
    
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    c = numeric_cmap((y+2)/4)
    
    axes[1, 1].scatter(x, y, c=c)


    for i in range(10):
        x = np.linspace(0, 1)
        y = x + np.random.normal(scale=0.1, size=50) + i
        
        axes[1, 2].plot(x, y, color=numeric_cmap(i/9), lw=3)
        
    fig.set_size_inches(12, 8)
    plt.show()