# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:27:10 2018

@author: Mike Staddon

Better plotting
"""
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 12}	

matplotlib.rc('font', **font)

from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np

from scipy.stats import binned_statistic

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cross_validation import train_test_split

import ml.stats

### Colours

bgCol = (1.0, 1.0, 1.0)
spineCol = (0.8, 0.8, 0.8)
color_1 = '#3B68B9' # Blue
color_2 = '#FF882B' # Orange
color_3 = '#F45B69' # Redish

def NumericColor(v, vmin=0, vmax=1, cmap=None):
    return matplotlib.cm.viridis((v - vmin)/(vmax - vmin))


def CategoricColor(i, n, cmap=None):
    return matplotlib.cm.tab10(i/10)


### Data transformation

def SelectCategories(data, column, categories=None, include_others=False):
    """ Select certain categories of one column
    Parameters
    ----------
    data: pandas dataframe
        Contains the data to be plotted
    categories: list
        list of categories to include, if None, include all
    include_others: bool
        if categories is provided, group the rest into an other category
    """
    
    # Avoid editting slice
    df = data[column].copy()
    df.astype(str)
    
    if categories is None:
        return df, [None]
    
    # Only create include_other if necessary
    if set(categories) == set(data[column].unique()):
        return df, categories
    
    if include_others:
        df[~df.isin(categories)] = 'Other'
        categories += ['Other']
    
    return df, categories


def GetColorColumn(data, color=None,
                   color_categoric=False, categories=None, include_others=False):
    """ Make a color column and categories
    Parameters
    ----------
    data: pandas dataframe
        Contains the data to be plotted
    color: str
        feature to color points, by default None
    color_categoric: bool
        true if color is a category, by default False
    categories: list
        list of categories to include, if None, include all
    include_others: bool
        if categories is provided, group the rest into an other category
    """

    if color is not None:
        df = data[color]
    else:
        return None, [None]
        
    if color is None or not color_categoric:
        color_cats = [None]
    else:
        df, color_cats = SelectCategories(data, color,
          categories=categories, include_others=include_others)
        
    return df, color_cats


def GetNumericPlotVariables(data, x, y=None, size=None,
                            color=None, color_categoric=False, color_categories=None,
                            fig=None, ax=None):
    
    # Avoid issues when x and y are the same, and allows modification
    df = pd.DataFrame()
    df['x'] = data[x]
    
    if y is not None:
        df['y'] = data[y]
        
    if size is not None:
        df['size'] = data[size]
    
    df['c'], color_cats = GetColorColumn(data, color=color,
                                         color_categoric=color_categoric,
                                         categories=color_categories)

    if fig is None:
        fig, ax = plt.subplot()
        
    if ax is None:
        ax = fig.add_subplot(111)
        
    return df, color_cats, fig, ax


def GetFactorVariables(data, x, y=None, x_options=None,
                       color=None, color_categoric=False, color_categories=None,
                       fig=None, ax=None):
    # Initialize variables for factor plots ie cat vs num

    # Avoid issues when x and y are the same, and allows modification
    df = pd.DataFrame()
    df['x'] = data[x]
    df['y'] = data[y]

    df['c'], color_cats = GetColorColumn(data, color=color,
                                         color_categoric=color_categoric,
                                         categories=color_categories)

    if fig is None:
        fig, axes = plt.subplots()
        
    bin_edges = None
    if x_options is None:
        cats = sorted(df['x'].unique())
        
    elif x_options['dtype'] == 'categoric':
        if 'categories' in x_options:
            cats = x_options['categories']
        else:
            cats = None
    
        if cats is None:
            cats = sorted(df['x'].unique())
        
    elif x_options['dtype'] == 'numeric':
        # Assign bin number to each value
        if 'bins' in x_options:
            bins = x_options['bins']
        else:
            bins = 10
            
        bins = bins if bins else 10
        
        stats, bin_edges, bin_num = binned_statistic(df['x'], df['x'], statistic='count', bins=bins)
        df['x'] = bin_num
        cats = [i+1 for i in range(max(bin_num))]
        
    if x_options is None or x_options['dtype'] == 'categoric':
        cat_locs = [i for i in range(len(cats))]
    elif x_options['dtype'] == 'numeric':
        cat_locs = (bin_edges[1:] + bin_edges[:-1])/2

    return df, cats, color_cats, cat_locs, fig, ax

### Styling

def MakeAxes(xs, ys, color=None, fig=None):
    """Make axes grid with an additional thin axes for colorbar or legend
    """
    if fig is None:
        fig = plt.figure()

    fig.clf()

    ncols, widths = len(xs), [1] * len(xs)

    if color is not None:
        ncols += 1
        widths += [1/40 * len(xs)]

    axes = []
    row = []

    gs = matplotlib.gridspec.GridSpec(len(ys), ncols, width_ratios=widths)

    for j in range(len(ys)):
        row = []
        for i in range(len(xs)):
            sharey = row[0] if i > 0 else None
            row += [fig.add_subplot(gs[j, i], sharey=sharey)]

        axes += [row]

    axes = np.array(axes)  

    cax = None
    if color is not None:
        cax = fig.add_subplot(gs[0,-1])

    return fig, axes, cax


def MakeLegend(ax, ys, color, color_categoric, color_cats=None, color_min=None, color_max=None):
    if color is None:
        return
    
    if color_categoric:
        # Legend
        handles = [matplotlib.lines.Line2D([0], [0],
                                           marker='o', lw=0,
                                           color=CategoricColor(j, len(color_cats)),
                                           label=str(col)) for j, col in enumerate(color_cats)]
            
        ax.legend(handles=handles, frameon=False, title=str(color),
                  loc='upper left', bbox_to_anchor=(0, 1))

        ax.axis('off')
            
    elif not color_categoric and color is not None:
        # Colorbar
        norm = matplotlib.colors.Normalize(color_min, color_max)
        cb = matplotlib.colorbar.ColorbarBase(ax, norm=norm, label=str(color), drawedges=True)

        # Style spines
        cb.outline.set_edgecolor(spineCol)
        cb.outline.set_linewidth(0.5)
        cb.dividers.set_linewidth(0)
        
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=spineCol)
        
        
def StyleAxis(ax, lines=True):
    """ Make the axes pretty """
    # Grid lines
    ax.set_axisbelow(True)
    
    if lines:
        ax.grid(True)
        xgridlines, ygridlines = ax.get_xgridlines(), ax.get_ygridlines()
        
        for line in xgridlines:
            line.set_linestyle('-')
            line.set_color('w')
            
        for line in ygridlines:
            line.set_linestyle('-')
            line.set_linewidth(1)
            line.set_color(spineCol)
        
    # Spine color
    plt.setp(ax.spines.values(), color=spineCol)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=spineCol)
        

def FormatAxis(data, x, y, cats, x_options, ax, showx=True, showy=True):
    """ Apply proper labels to axes """
    
    # Labels
    if showx:
        ax.set_xlabel(x)

        # x axis takes numeric values - automatically labelled by mpl
        if cats is None or x_options is not None and x_options['dtype'] == 'numeric':
            pass
        else:
            ax.set_xticks([j for j in range(len(cats))])
            ax.set_xticklabels([cat for cat in cats])
            for tick in ax.get_xticklabels():
                tick.set_rotation(-45)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)

    if showy:
        if y is None:
            y_label = 'Frequency'
        else:
            y_label = y
            
        ax.set_ylabel(y_label)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

    StyleAxis(ax)


### Numeric vs Numeric
def PairPlot(data, xs, color=None, color_categoric=False, color_categories=None, 
             size=None, lims=None,
             fig=None):
    """ Pair plot of all x. Returns scatter for different x, histograms for
    same.
    Parameters
    ----------
    data: pandas dataframe
        Contains the data to be plotted
    xs: str or list
        features to be plotted
    color: str
        feature to color points, by default None
    color_categoric: bool
        true if color is a category, by default False
    color_categories: dict
        list of color categories to include
    size: str
        feature for size of dots, by default None
    lims: list of tuples
        limits for each category
    fig: matplotlib figure
        figure to create subplots with, by default a new fig is created
    """
    
    if type(xs) is str:
        xs = [xs]
    
    fig, axes, cax = MakeAxes(xs, xs, color=color, fig=fig)
        
    if lims is None:
        lims = [[data[x].min()-0.05*(data[x].max()-data[x].min()),
                 data[x].max()+0.05*(data[x].max()-data[x].min())] for x in xs]
        
    for i, y in enumerate(xs):
        for j, x in enumerate(xs):
            ax = axes[i, j]
            # Only scatter off diagonal
            if i != j:
                ScatterPlot(data, x, y,
                            color=color,
                            color_categoric=color_categoric,
                            color_categories=color_categories,
                            size=size,
                            fig=fig, ax=ax)
            else:
                Histogram(data, x,
                          color=color,
                          color_categoric=color_categoric,
                          color_categories=color_categories,
                          lims=lims[i],
                          fig=fig, ax=ax)
            
    # Limits and labels
    for i, y in enumerate(xs):
        for j, x in enumerate(xs):
            ax = axes[i, j]
            
            FormatAxis(data, x, y, None, None, ax,
                       showx=(i == len(xs)-1), showy=(j == 0))
            
            ax.set_ylim(lims[i])

    if color is not None:
        
        _, color_cats = GetColorColumn(data, color=color, color_categoric=color_categoric,
                                       categories=color_categories)
        MakeLegend(cax, xs, color, color_categoric, color_cats=color_cats,
                   color_min=data[color].min(), color_max=data[color].max())
        
    fig.tight_layout()
    
    
def Histogram(data, x, color=None, color_categoric=None, color_categories=None,
              lims=None,
              fig=None, ax=None):
    
    if lims is None:
        lims = [0, len(data)]
    
    df, color_cats, fig, ax = GetNumericPlotVariables(data, x, y=None,
                                                      color=color,
                                                      color_categoric=color_categoric,
                                                      color_categories=color_categories,
                                                      fig=fig, ax=ax)
    
    # First pass to get maxes
    nbins = 10
    counts, bins = np.histogram(df['x'],
                                bins=nbins,
                                range=[df['x'].min(), df['x'].max()])
    
    max_count = max(counts)
    
    previous = lims[0]
    for k, col in enumerate(color_cats):
        # Subset data
        if len(color_cats) == 1 or not color_categoric:
            sub = df
        else:
            sub = df[df['c'] == col]
            
        counts, bins = np.histogram(sub['x'],
                                    bins=nbins,
                                    range=[df['x'].min(), df['x'].max()])
        
        # Scale counts with axis height
        counts = 0.9*counts.astype(np.float32)*(lims[1] - lims[0])/max_count
        
        # Colour by mean value in bin
        if not color_categoric and color is not None:
            c, _, _ = binned_statistic(sub['x'], sub['c'], 'mean', bins)
            c = NumericColor(c, df['c'].min(), df['c'].max())
        else:
            c = CategoricColor(k, len(color_cats)-1)

        width = bins[1]-bins[0]
        ax.bar(bins[:-1] + width/2, counts,
               width=width, bottom=previous, color=c,
               edgecolor=bgCol, lw=0.5)
        
        previous += counts
        
    FormatAxis(data, x, None, None, None, ax)
        
        
def ScatterPlot(data, x, y, color=None, color_categoric=None, color_categories=None,
                size=None,
                fig=None, ax=None):
    
    df, color_cats, fig, ax = GetNumericPlotVariables(data, x, y=y, size=size,
                                                      color=color,
                                                      color_categoric=color_categoric,
                                                      color_categories=color_categories,
                                                      fig=fig, ax=ax)
    
    for k, col in enumerate(color_cats):
        # Subset data
        c = None
        if len(color_cats) == 1 or not color_categoric:
            sub = df
            if color is not None:
                c = df['c']
        else:
            sub = df[df['c'] == col]
            c = CategoricColor(k, len(color_cats))

        if size is None:
            s = 64
        else:
            s = 64 * sub['size'] / df['size'].max()
            
        ax.scatter(sub['x'], sub['y'],
                   c=c, s=s, edgecolors=bgCol, lw=0.5)


### Numeric vs Categoric
def FactorPlot(data, xs=None, ys=None, styles=None, xs_options=None,
               color=None, color_categoric=False, color_categories=None,
               transposed=False,
               fig=None):        
    """ Plots categoric features against numeric, with different plot styles.
    
    Arguments
    
    ---------
    
    data: pandas dataframe
        Contains the data to be plotted
    xs: str or list
        categoric features as x values
    ys: str or list
        numeric features as y values
    styles: str or list
        type of each plot corresponding to ys
    xs_options: dictionary, optional
        contains options for each features eg which categories to use, or 
        whether a numeric value should be binned using a histogram
    color: str, optional
        feature to color points, by default None
    color_categoric: bool, optional
        true if color is a category, by default False
    fig: matplotlib figure, optional
        figure to create subplots with, by default a new fig is created
        
    ---------
    TO DO:
        Med priority:
            - implement color options eg which cats, vmin, vmax
            - handle large number of categories?
            - transpose
        Low priority:
            - implement custom colormaps
        
    """
    
    if type(xs) is str:
        xs = [xs]
    
    if xs_options is None:
        xs_options = {}
    
    fig, axes, cax = MakeAxes(xs, ys, color=color, fig=fig)
        
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ax = axes[i, j]
            
            x_options = xs_options[x] if x in xs_options else None
            
            if styles[i] == 'jitter':
                JitterPlot(data, x, y, x_options=x_options,
                           color=color,
                           color_categoric=color_categoric, color_categories=color_categories,
                           fig=fig, ax=ax,
                           showx=(i == len(ys)-1), showy=(j == 0))
                
            elif styles[i] == 'box':
                BoxPlot(data, x, y, x_options=x_options,
                        color=color,
                        color_categoric=color_categoric, color_categories=color_categories,
                        fig=fig, ax=ax,
                        showx=(i == len(ys)-1), showy=(j == 0))
                
            elif styles[i] == 'mean' or styles[i] == 'total':
                StatPlot(data, x, y, x_options=x_options,
                         stat=styles[i],
                         color=color,
                         color_categoric=color_categoric, color_categories=color_categories,
                         fig=fig, ax=ax,
                         showx=(i == len(ys)-1), showy=(j == 0))
                
            elif styles[i] == 'frequency':
                FrequencyPlot(data, x, y=None, x_options=x_options,
                              color=color,
                              color_categoric=color_categoric, color_categories=color_categories,
                              fig=fig, ax=ax,
                              showx=(i == len(ys)-1), showy=(j == 0))
                
                
    if color is not None:
        _, color_cats = GetColorColumn(data, color=color,
                                   color_categoric=color_categoric,
                                   categories=color_categories)
        
        MakeLegend(cax, xs, color, color_categoric, color_cats=color_cats,
                   color_min=data[color].min(), color_max=data[color].max())
        
    fig.tight_layout()
    


        
    
    
def JitterPlot(data, x, y, x_options=None,
               color=None, color_categoric=False, color_categories=None,
               fig=None, ax=None,
               showx=True, showy=True):
    
    (df,
     cats,
     color_cats,
     cat_locs,
     fig,
     ax) = GetFactorVariables(data, x, y, x_options=x_options,
                              color=color,
                              color_categoric=color_categoric,
                              color_categories=color_categories,
                              fig=fig, ax=ax)
        
    # Width of each category
    width = 0.8/len(color_cats) * (cat_locs[1] - cat_locs[0])
    
    # Subset data
    for i, col in enumerate(color_cats):
        if len(color_cats) == 1 or not color_categoric:
            sub = df
            if color is not None:
                c = sub['c']
        else:
            sub = df[df['c'] == col]
            c = CategoricColor(i, len(color_cats))
                
        base_x, base_y = [], []
        
        for j, cat in enumerate(cats):
            if len(cats) > 1:
                sub2 = sub[sub['x'] == cat]
            else:
                sub2 = sub

            # Get x and y
            base_y += list(sub2['y'])
            base_x += [cat_locs[j] + i*width - 0.4 + 0.4/len(color_cats)] * len(sub2)
            
        # Add jitter
        base_x = np.array(base_x) + np.random.uniform(low=-width/3, high=width/3, size=(len(base_x),))
        
        ax.scatter(base_x, base_y, c=c, lw=0.5, edgecolor=bgCol)
        
    FormatAxis(data, x, y, cats, x_options, ax, showx=showx, showy=showy)
        
        
def BoxPlot(data, x, y, x_options=None,
            color=None, color_categoric=False, color_categories=None,
            fig=None, ax=None,
            showx=True, showy=True):
    
    (df,
     cats,
     color_cats,
     cat_locs,
     fig,
     ax) = GetFactorVariables(data, x, y, x_options=x_options,
                              color=color,
                              color_categoric=color_categoric,
                              color_categories=color_categories,
                              fig=fig, ax=ax)

    # Width of each category
    width = 0.8/len(color_cats) * (cat_locs[1] - cat_locs[0])
        
    # Subset data
    for i, col in enumerate(color_cats):
        c = None
        if len(color_cats) == 1 or not color_categoric:
            sub1 = df
            if color is not None:
                c = np.array([sub1[sub1['x'] == cat]['c'].mean() for cat in cats])
                c = NumericColor(c, sub1['c'].min(), sub1['c'].max())
        else:
            sub1 = df[df['c'] == col]
            c = CategoricColor(i, len(color_cats))

        ten, q1, q2, q3, ninety = [], [], [], [], []
        
        for cat in cats:
            if len(cats) > 1:
                sub2 = sub1[sub1['x'] == cat]
            else:
                sub2 = sub1

            # Get mean and sd
            quants = sub2['y'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            ten.append(quants[0.10])
            q1.append(quants[0.25])
            q2.append(quants[0.50])
            q3.append(quants[0.75])
            ninety.append(quants[0.90])

        xlocs = [cat_locs[j] + i*width - 0.4 + 0.4/len(color_cats) for j in range(len(cats))]
        
        ten, q1, q2, q3, ninety = np.array(ten), np.array(q1), np.array(q2), np.array(q3),np.array(ninety)
        
        # Draw whiskers
        for j in range(len(xlocs)):
            ax.plot([xlocs[j]]*2, [ten[j], ninety[j]], color='k', zorder=-1)
            ax.plot([xlocs[j]-width/4, xlocs[j]+width/4],
                    [ten[j], ten[j]], color='k', zorder=-1)
            ax.plot([xlocs[j]-width/4, xlocs[j]+width/4],
                    [ninety[j], ninety[j]], color='k', zorder=-1)
            
        # Draw IQR
        ax.bar(xlocs, q3-q1, bottom=q1,
               color=c, width=width, edgecolor='k', lw=1.5)
        
        # Draw median
        for j in range(len(xlocs)):
            ax.plot([xlocs[j]-width/2, xlocs[j]+width/2], [q2[j], q2[j]], color='k')
        
    FormatAxis(data, x, y, cats, x_options, ax, showx=showx, showy=showy)
    

def StatPlot(data, x, y,
             x_options=None, stat=None,
             color=None, color_categoric=False, color_categories=None,
             fig=None, ax=None,
             showx=True, showy=True):

    stat = stat if stat else 'total'
    
    (df,
     cats,
     color_cats,
     cat_locs,
     fig,
     ax) = GetFactorVariables(data, x, y, x_options=x_options,
                              color=color,
                              color_categoric=color_categoric,
                              color_categories=color_categories,
                              fig=fig, ax=ax)

    # Width of each category
    width = 0.8/len(color_cats) * (cat_locs[1] - cat_locs[0])

    # Subset data
    for i, col in enumerate(color_cats):
        c = None
        if len(color_cats) == 1 or not color_categoric:
            sub1 = df
            if color is not None:
                c = np.array([sub1[sub1['x'] == cat]['c'].mean() for cat in cats])
                c = NumericColor(c, sub1['c'].min(), sub1['c'].max())
        else:
            sub1 = df[df['c'] == col]
            c = CategoricColor(i, len(color_cats))

        stat1, stat2 = [], []
        
        for cat in cats:
            if len(cats) > 1:
                sub2 = sub1[sub1['x'] == cat]
            else:
                sub2 = sub1

            if stat == 'total':
                stat1.append(sub2['y'].sum())
                stat2 = None
            elif stat == 'mean':
                stat1.append(sub2['y'].mean())
                stat2.append(sub2['y'].std())
        
        xlocs = [cat_locs[j] + i*width - 0.4 + 0.4/len(color_cats) for j in range(len(cats))]
            
        # Draw bars
        ax.bar(xlocs, stat1, yerr=stat2,
               color=c, width=width, edgecolor=bgCol, lw=0.5)
        
    FormatAxis(data, x, y, cats, x_options, ax, showx=showx, showy=showy)
    
    
# Categoric vs None
def FrequencyPlot(data, x, y=None,
                  x_options=None,
                  color=None, color_categoric=False, color_categories=None,
                  fig=None, ax=None,
                  showx=True, showy=True):
    
    """Draw a plot counting each category and colour"""

    # Put in x as y for a hacky solution
    df, cats, color_cats, cat_locs, fig, ax = GetFactorVariables(data, x, x, x_options=x_options,
                                                                 color=color, color_categoric=color_categoric,
                                                                 fig=fig, ax=ax)
    
    # Width of each category
    width = 0.8 * (cat_locs[1] - cat_locs[0])

    # Subset data
    previous = 0
    for i, col in enumerate(color_cats):
        c = None
        if len(color_cats) == 1 or not color_categoric:
            sub1 = df
            if color is not None:
                c = np.array([sub1[sub1['x'] == cat]['c'].mean() for cat in cats])
                c = NumericColor(c, sub1['c'].min(), sub1['c'].max())
        else:
            sub1 = df[df['c'] == col]
            c = CategoricColor(i, len(color_cats))

        counts = []
        
        for cat in cats:
            if len(cats) > 1:
                sub2 = sub1[sub1['x'] == cat]
            else:
                sub2 = sub1

            counts.append(len(sub2['y']))
        
        xlocs = cat_locs
        counts = np.array(counts)
            
        ax.bar(xlocs, counts,
               width=width, bottom=previous, color=c,
               edgecolor=bgCol, lw=0.5)
            
        previous += counts
        
    FormatAxis(data, x, None, cats, x_options, ax, showx=showx, showy=showy)
    
    
### Text
def GenerateWordCloud(data, column, color, category=None,
                      fig=None, ax=None):
    
    x, y = data[column].astype(str), data[color]
    
    if category is not None:
        y = y == category
    
    #Set nan values to 'missing' so groupby works with them
    x = x.map(lambda x: '#missing' if x is np.nan else x)
    
    counter = CountVectorizer(min_df=0.001)
    counted = counter.fit_transform(x)
    
    res = pd.DataFrame(columns=['Word', 'Count', 'Value'])
    res['Word'] = sorted(counter.vocabulary_.keys())
    res['Count'] = counted.sum(axis=0).transpose()
    res['Value']  = counted.multiply(y.values.reshape(len(y),1)).sum(axis=0).transpose()
    res['Value'] = res['Value']/res['Count']
    
    WordCloud(res['Word'], res['Count'], res['Value'],
              fig=fig, ax=ax)


def WordCloud(words, sizes, colors, minfontsize=4, maxfontsize=48, max_words=100,
              vmin=None, vmax=None,
              fig=None, ax=None):
    
    """
    Arguments
        words: list of words to plot
        sizes: size of word, could be frequencies
        colors: numeric values determining color
        minfontsize: fontsize of smallest word
        maxfontsize: fontsize of largest word
        vmin: min value for colorbar
        vmax: max value for colorbar
        fig: mpl figure
        ax: mpl axis
    """
    
    # Check no collisions and within plot
    def BadTextPosition(bbox, bboxes):
        if bbox.count_overlaps(bboxes) > 0:
            return True
        
        # Break on out of bounds
        left, right, bottom, top = bbox.xmin, bbox.xmin + bbox.width, bbox.ymin, bbox.ymin + bbox.height
        inv = ax.transData.inverted()
        left, top = inv.transform([left, top])
        right, bottom = inv.transform([right, bottom])

        if left < 0 or right > 1 or top > 1 or bottom < 0:
            return True
        
        return False
        

    if fig is None or ax is None:
        fig, ax = plt.subplots()
        
    cdict = {'red': ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
    
            'green': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),

            'blue': ((0.0, 1.0, 1.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
         
    cmap = matplotlib.colors.LinearSegmentedColormap('BlueRed1', cdict)
        
    # Sort words by size
    words = [w for _,w in sorted(zip(sizes, words))][::-1]
    colors = [c for _,c in sorted(zip(sizes, colors))][::-1]
    sizes = sorted(sizes)[::-1]
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    bboxes = []
    
    renderer = fig.canvas.get_renderer()
    
    # Begin plotting first word
    r, theta_0, theta_mult = 0, 0, 1
    for w, c, s in zip(words, colors, sizes):
        r, theta_0, theta_mult = 0, np.random.uniform(0, np.pi * 2), np.random.choice([-1, 1])
        if len(bboxes) == max_words:
            return
        
        x, y = [0.5, 0.5]

        color = (c-min(colors))/(max(colors)-min(colors))
        size = (s/max(sizes))**0.5 * (maxfontsize-minfontsize) + minfontsize
        
        txt = ax.text(x, y, w.capitalize(), weight='bold', color=cmap(color), fontsize=size,
                      va='center', ha='center', transform=ax.transData)
        
        bbox = txt.get_window_extent(renderer)
        
        if len(bboxes) == 0:
            # Reduce maxsize so that we fit at least one word in
            while BadTextPosition(bbox, bboxes):
                maxfontsize -= 1
                txt.set_fontsize(maxfontsize)
                bbox = txt.get_window_extent(renderer)
        else:
            # Find next available space in a spiral
            while BadTextPosition(bbox, bboxes) and r < 0.707:
               
                r += 0.001
                theta = theta_mult * np.pi * r**0.5 * 33 + theta_0
                x, y = [0.5 + r**0.5 * np.cos(theta), 0.5 + r**0.5 * np.sin(theta)]
                txt.set_position((x, y))
                bbox = txt.get_window_extent(renderer)
            
        # Remove text and stop
        if r >= 0.707:
            txt.set_visible(False)
        else:
            bboxes.append(bbox)
            
            
### Machine learning evaluation plots

def ConfusionMatrix(y, ypred,
                    classes=None,
                    fig=None, ax=None):
    
    # Custom cmap
    cdict = {'red': ((0.0, 43/255, 43/255),
                     (0.5, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
    
             'green': ((0.0, 83/255, 83/255),
                      (0.5, 1.0, 1.0),
                      (1.0, 136/255, 136/255)),
                     
             'blue': ((0.0, 169/255, 169/255),
                     (0.5, 1.0, 1.0),
                     (1.0, 43/255, 43/255))}
    
            
    cmap = matplotlib.colors.LinearSegmentedColormap('BlueWhiteOrange', cdict)
    
    if fig is None:
        fig = plt.figure()
        
    if ax is None:
        ax = fig.add_subplot(111)
        
    if classes is None:
        classes = list(set(y))
    
    cm = confusion_matrix(y, ypred)

    # Take percentages
    cm = cm / np.sum(cm, axis=1)[:, np.newaxis]
    
    #Highlight succsesful classifications in blue, not in orange
    for i in range(cm.shape[0]):
        cm[i, i] = -cm[i, i]
            
    lim = max(cm.max(), -cm.min())
    ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-lim, vmax=lim)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add labels to each point
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, '{:.0f}%'.format(100*abs(cm[i, j])),
                    horizontalalignment="center", color="black")

    #a.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    fig.autofmt_xdate()
    fig.tight_layout()
    
    #Background
    ax.patch.set_alpha(1)
    
    # Spine color
    plt.setp(ax.spines.values(), color=spineCol)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=spineCol)
    
    fig.tight_layout()
    
    
def ROCCurve(y, yprob, classes=None, fig=None, ax=None):
    """ Plot the ROC Curve and calculate AUC score 
    
    Parameters:
        y: array
            true observations
            
        yprob: array
            predicted probabilities
            
        classes: list
            list of classes predicted
            
        fig: mpl figure
            figure to plot on, if None a new one is made
            
        ax: mpl axis
            axis to plot on, if None a new one is made
            
        """
    
    if fig is None:
        fig = plt.figure()
        
    if ax is None:
        ax = fig.add_subplot(111)

    if classes is None:
        classes = list(set(y))
        
    for index, cat in enumerate(classes):
        #Get false positive and true positive rates
        fpr, tpr, _ = roc_curve(y == cat, yprob[:, index])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC
        ax.plot(fpr, tpr, lw=3, label='{:} (Area = {:.3f})'.format(cat, roc_auc))
    #    ax.fill_between(fpr, tpr, color='C0', alpha=0.33)
    
    ax.plot([0, 1], [0, 1], color=spineCol, lw=3, linestyle='--')

    ax.legend(fontsize=10,
              frameon=False,
              loc='center left',
              bbox_to_anchor=(1.05, 0.5))
    
#    FormatChart(fig, ax, xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    ax.set_aspect('equal')
    
    #Background
    ax.patch.set_alpha(1)
    
    # Spine color
    plt.setp(ax.spines.values(), color=spineCol)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=spineCol)

    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    fig.tight_layout()
    
    
def TrainingPlot(time, scores, scorer=None, cv=True, fig=None, ax=None):
    """ Plots score vs time, and best score for a machine learning estimator

    Parameters:
        time: list
            list of time for training
        scores: list
            list of scores of training
        scorer: string, optional
            metric used in training
        cv: boolean, optional
            if these are cv scores, instead of test scores
        fig: mpl figure, optional
            default None
        ax: mpl axis, optional
            default None
    """
    
    
    def FormatScore(score, scorer):
        # Most maxmimise the negative of the loss
        if scorer != 'r2':
            score = abs(score)
            
#        if metric == 'neg_mean_squared_error':
#            score = score**0.5
        
        return score
    
    if fig is None:
        fig = plt.figure()
        
    if ax is None:
        ax = fig.add_subplot(111)

    if len(time) > 0:
        # Best results
        min_loss, min_time = [], []
        
        if max(time) > 300:
            time = np.array(time)/60
            xlabel = 'Training Time (mins)'
        else:
            xlabel = 'Training Time (s)'
            
        for i in range(len(scores)):
            if len(min_loss) == 0 or scores[i] > min_loss[-1]:
                min_loss.append(scores[i])
                min_time.append(time[i])
                
        min_loss.append(min_loss[-1])
        min_time.append(time[-1])
    
        min_loss = [FormatScore(s, scorer) for s in min_loss]
        scores = [FormatScore(s, scorer) for s in scores]
        
        ax.scatter(time, scores, color=color_2, label='Run Score', zorder=1)
        ax.plot(min_time, min_loss, lw=3, color=color_1, label='Best Score', zorder=2,
                marker='o')
    
        # Manually make legend
        leg = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', 
                    frameon=False, markerscale=0, handlelength=0)
        
        i = 0
        for text in leg.get_texts():
            text.set_color(['C0', 'C1'][i])
            i += 1
        
        # Set appropriate limits
        if scorer in ['r2', 'accuracy']:
            ymax = 1.05
        else:
            ymax = None
            
        ymin = 0
        
        ax.set_xlim([0, time[-1]*1.05])
        ax.set_ylim([ymin, ymax])
    else:
        # No models run yet
        ax.set_xticks([])
        ax.set_yticks([])
    
        xlabel = 'Training Time (s)'
    
    if cv:
        split = 'CV '
    else:
        split = 'Test '
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(split + 'Score')
    
    StyleAxis(ax)
    
    fig.tight_layout()
    
    
def FeatureDependence(X, y, feature, dtype,
                      bins=None, model=None, cat=None, target=None,
                      ylims=None,
                      fig=None, ax=None):
    """ Plot how the mean value or proportion of y changes with feature 
    TO DO: rename as x and y in to be consistent
    
    Arguements:
        
    """
    
    means, counts, bins = ml.stats.feature_dependence(X, y, feature, dtype, model=model, cat=cat, bins=bins)
    
    if target is None:
        target = ''
        
        if cat is None:
            ylabel = 'Average ' + target
        else:
            ylabel = 'Proportion ' + target + ': ' + str(cat)
        
    if dtype == 'numeric':
        fig, axes, cax = MakeAxes([feature], ['Frequency'], color=True, fig=fig)
        
        ax = axes[0,0]
        
        # Histogram of counts
        width = bins[1] - bins[0]
        ax.bar(bins, counts, width=width, color=color_1,
               edgecolor=bgCol, lw=0.5, zorder=10)
        
        # Average line plots
        ax2 = ax.twinx()
        
        bins, means = bins[counts != 0], means[counts != 0]
        
        ax2.plot(bins, means['y'], color=color_2, lw=3,
                 marker='x', ms=9, mew=3)
        
        if model is not None:
            ax2.plot(bins, means['ypred'], color=color_3, lw=3, 
                    marker='+', ms=9, mew=3)
            
            
        # A dummy axis to draw grid lines underneath!        
        # Third axis is just for grid lines, can't draw averages over counts over lines!
        ax3 = ax.twinx()
        ax3.set_zorder(-2)
        ax3.patch.set_alpha(0)
        
        ax.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        
        ax2.set_yticks([])
        ax2.set_ylim(ylims)

        ax3.set_ylabel(ylabel)
        ax3.set_ylim(ylims)
        
        # Style axes
        StyleAxis(ax, lines=False)
        StyleAxis(ax2, lines=False)
        
        StyleAxis(ax3)
        
        # Make legend
        handles = [matplotlib.lines.Line2D([0], [0], lw=3,
                                           color=['C1', 'C3'][j],
                                           label=['Actual', 'Predicted'][j]) for j in range(2)]
    
        cax.legend(handles=handles, frameon=False,
                   loc='upper left', bbox_to_anchor=(0, 1))
        
        cax.axis('off')
        
        fig.tight_layout()
        
    else:
        # TO DO: improve, although at the moment its not used
        if dtype == 'categoric':
            name = 'Category'
        elif dtype == 'text':
            name = 'Word'
            
        # Sort by values, show up to top 25
        means['count'] = counts
        means[name] = bins
        means[name] = means[name].apply(str)
        means = means.sort_values(by='count', ascending=False).iloc[:25, :]
        means = means.sort_values(by=name, ascending=False)
        
        x = range(len(means))
            
        fig, axes, _ = MakeAxes(['Count', 'Actual', 'Predicted'], [name])
        
        axes[0, 0].barh(x, means['count'].values)
        axes[0, 1].barh(x, means['y'].values)
        axes[0, 2].barh(x, means['ypred'].values)

        axes[0, 0].set_yticks(x)
        axes[0, 0].set_yticklabels(means[name].values)

        plt.setp(axes[0, 1].get_yticklabels(), visible=False)
        plt.setp(axes[0, 2].get_yticklabels(), visible=False)
        

if __name__ == '__main__':
#    data = pd.read_csv('D:/Kaggle/Titanic/train.csv')
    
#    PairPlot(data, ['Fare', 'Age'], color='Pclass', color_categoric=True)
#    plt.gcf().set_size_inches(8, 8)
#    plt.show()
#    
#    xopts = {'Embarked': {'type': 'categoric',
#                          'categories': ['C', 'S']},
#             'Fare': {'type': 'numeric', 'bins': 20}}
#    
#    FactorPlot(data, ['Embarked', 'Fare'], ['Age', 'Fare', None], ['jitter', 'box', 'frequency'],
#               xs_options=xopts,
#               color='Survived', color_categoric=True)
#    plt.gcf().set_size_inches(8, 8)
#    plt.show()
    
#    data = pd.read_csv('D:/Kaggle/House Prices/train.csv')
#
#    PairPlot(data, ['1stFlrSF', '2ndFlrSF'], color='OverallQual',
#             color_categoric=True, color_categories=[1, 2, 3, 4, 5])
#    plt.gcf().set_size_inches(9,6)
#    plt.gcf().tight_layout()
#    plt.show()
#    
#    xopts = {'1stFlrSF': {'type': 'numeric', 'bins': 20}}
#    FactorPlot(data, ['OverallCond', 'OverallQual', '1stFlrSF'], ['SalePrice', '1stFlrSF', 'SalePrice'],
#               ['jitter', 'box', 'mean'], xs_options=xopts,
#               color='SalePrice')
#    plt.gcf().set_size_inches(12, 12)
#    plt.gcf().tight_layout()
#    plt.show()
    
#    xopts = {'1stFlrSF': {'type': 'numeric', 'bins': 20}}
#    col_cats = [1, 2, 3, 4, 5]
#    FactorPlot(data, xs=['OverallCond', '1stFlrSF'], ys=['SalePrice', '1stFlrSF'],
#               types=['jitter', 'box', 'mean'], xs_options=xopts,
#               color='OverallQual', color_categoric=True, color_categories=col_cats)
#    plt.gcf().set_size_inches(12, 12)
#    plt.gcf().tight_layout()
#    plt.show()
#
#    ConfusionMatrix([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3], [0, 0, 1, 1, 2, 0, 2, 3, 3, 3, 2])
#    plt.show()
#    
#    from sklearn.datasets import load_iris
#    from sklearn.linear_model import LogisticRegression
#    
#    X, y = load_iris(True)
#    mod = LogisticRegression()
#    mod.fit(X, y)
#    probs = mod.predict_proba(X)
#
#    ROCCurve(y, probs)
#    plt.show()

#    n_times = 20
#    time = [0]
#    for i in range(n_times):
#        time.append(time[-1] + np.random.exponential(5))
#        
#    scores = [np.random.exponential(1) for t in time]
#    TrainingPlot(time, scores)
#    plt.show()
#    
#    TrainingPlot([], [])
#    plt.show()

    from sklearn.linear_model import LinearRegression
    df = pd.DataFrame()
    df['x'] = np.random.poisson(size=1000)
    df['z'] = np.random.normal(size=1000)
    
    df['y'] = df['x'] + np.random.normal(scale=1, size=1000)
    
    mod = LinearRegression()
    mod.fit(df[['x','z']], df['y'])
    
    FeatureDependence(df[['x','z']], df['y'], 'x', 'numeric', model=mod, bins=5)
#    plt.gcf().set_size_inches(8, 5)
    plt.show()
    
#    FeatureDependence(df[['x','z']], df['y'], 'x', 'categoric', model=mod)
#    plt.show()