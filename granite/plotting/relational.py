# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:41:00 2018

@author: Mike Staddon
"""

import pandas as pd
import numpy as np
from scipy.stats import binned_statistic

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import matplotlib

from .colors import numeric_cmap, categoric_cmap, colors


def categorical_stats(data, x, y=None, color=None, stats=None):
    """
    Get groupby statistics from a pandas dataframe
    
    Arguments
    ---------
    
    data: dataFrame
        raw data
        
    x: str
        column to group stats by
    
    y: str
        column to get stats of
        
    color: str
        subgroups column, as in plotting
        
    stats: None, 'quartiles', or other
        the statistics to get.
        If None, get counts.
        If 'quartiles' get quartiles
        Other stats should be pandas groupby.agg functions eg 'mean', or
        ['mean', 'sd']
        
    Returns
    -------
    
    statstics: dataFrame
        stats with columns, x, color, and statistics columns
    
    """
    
    df = pd.DataFrame()
    df['x'] = data[x]
    by = ['x']
    
    if y is not None:
        df['y'] = data[y]
    else:
        df['count'] = 1
        
    if color is not None:
        # Make categorical so we see all sub categories
        df['color'] = data[color].astype('category')
        by = ['x', 'color']
        
    g = df.groupby(by)
    
    if stats is None or y is None:
        # Return counts - need a dummy column for this
        return g.count()['count'].reset_index()
    elif stats == 'quantiles':
        # Return 10th, quartiles, and 90th
        return g.quantile([0.1, 0.25, 0.5, 0.75, 0.9])['y'].reset_index()
    else:
        return g.agg(stats)['y'].reset_index()


def text_stats(data, x, y=None, color=None, stats=None, max_features=25):
    """
    Get count and mean stats by analysing text
    """
    
    # TO DO: colors not in correct order!
    
    df = pd.DataFrame()
    df['x'] = data[x]
    cols = [None]
    
    if y is not None:
        df['y'] = data[y]
        
    if color is not None:
        # Make categorical so we see all sub categories
        df['color'] = data[color].astype('category')
        cols = sorted(list(df['color'].unique()))
      
    # Count word frequency by row
    counter = CountVectorizer(max_features=max_features, stop_words='english')
    counter.fit(df['x'])

    # Vocabulary maps words to columns, unmap to get corresponding words
    kvs = counter.vocabulary_
    words = sorted(list(kvs.keys()), key=lambda k: kvs[k])
    
    # Get stats for each word and colour
    results = pd.DataFrame()
    for col in cols:

        sub_results = pd.DataFrame()
        if col is None:
            sub = df
        else:
            sub = df[df['color'] == col]
            
        sub_results['x'] = words
        sub_results['color'] = col
        
        # Rows are observations, columns are words, and values are counts
        counted = counter.transform(sub['x'])
        sub_results['count'] = counted.sum(axis=0).A[0]
        
        if y is not None:
            sub_results['mean'] = counted.multiply(sub['y'].values.reshape(-1, 1)).sum(axis=0).A[0] / sub_results['count']

        results = results.append(sub_results, ignore_index=True)
        
    # Sort by color then word
    results = results.sort_values(by=['x', 'color']).reset_index()

    return results


class _RelationalPlotter():
    def __init__(self, data, x, y=None,
                 x_type=None, x_bins=None, x_order=None,
                 color=None, color_type=None, color_order=None,
                 fig=None, ax=None):
        """ A base plotting class that handles plotting of y against x
        
        Parameters
        ----------
        
        data: pandas Dataframe
            container of the data
            
        x, y: str
            column names to plot
            
        x_categoric: bool
            if True, treat x as a category. If data[x] is a string then it
            will be set to categoric automatically
            
        x_bins: int
            if not None, bin values of x, to get statistics within each bin
            
        x_order: list
            list of categories to use if x_categoric. If None, use all
            
        color: str
            column name used to color values
            
        color_order: list
            list of categories to use if x_categoric. If None, use all
        """

        self.data = data
        self.x = x
        self.y = y

        # x options
        if x_type is None:
            if self.check_column_numeric(data, x):
                x_type = 'numeric'
            else:
                x_type = 'categoric'
            
        self.x_type = x_type
        self.x_bins = x_bins
        
        if self.x_type == 'categoric':
            if x_order is None:
                self.x_order = self.get_top_categories(data, x, 25)
            else:
                self.x_order = x_order
        else:
            self.x_order = None


        # color options
        if color is not None and color_type is None:
            if self.check_column_numeric(data, color):
                color_type = 'discrete'
            else:
                color_type = 'categoric'
            
        self.color = color
        self.color_type = color_type
        
        if self.color is not None:
            if color_order is not None:
                self.color_order = color_order
            else:
                self.color_order = self.get_top_categories(data, color, 10)
        else:
            self.color_order = [None]
        
        self.fig, self.ax = self.make_axis(fig, ax)
        
        self.plot_data = self.get_plot_data(self.data, self.x, y=self.y,
                                            x_type=self.x_type, x_bins=self.x_bins, x_order=self.x_order,
                                            color=self.color, color_order=self.color_order)
            
            
    def get_top_categories(self, data, column, max_n=25):
        counts = data[column].value_counts()
        
        # Select most frequent
        counts = counts.iloc[:min(len(counts), max_n)]
        
        # Sort categories
        return sorted(list(counts.index))

        
    def make_axis(self, fig=None, ax=None):
        """ Create figure and axis if needed"""
        if fig is None:
            fig = plt.gcf()
            ax = None
            
        if ax is None:
            ax = plt.gca()
            
        return fig, ax
    
    
    def make_legend(self):
        if self.color is None:
            return
            
        # Legend
        handles = [matplotlib.lines.Line2D([0], [0],
                                           marker='o', lw=0,
                                           color=self.color_map[cat],
                                           label=str(cat)) for cat in self.color_map]
            
        self.ax.legend(handles=handles, frameon=False, title=str(self.color),
                       loc='upper left', bbox_to_anchor=(1, 1))                

    
    def check_column_numeric(self, data, column):
        """ Returns true if a column is numeric """
        return pd.api.types.is_numeric_dtype(data[column])


    def subset_column(self, data, column, categoric=False, categories=None):
        """ Select chosen categories, set others to nan """
        
        # Update?: can replace this with pd.isin() or a similar function
        if not categoric:
            categoric = not self.check_column_numeric(data, column)
            
        if not categoric:
            return data[column]
        
        # Remove unused categories
        if categories is None:
            categories = np.unique(data[column])

        c = []
        for d in data[column]:
            if d in categories:
                c.append(d)
            else:
                c.append(np.nan)
            
        return c


    def get_color_mapping(self, c, cmap=None, vmin=None, vmax=None):
        """ Convert color column into rgba codes """
        if self.color is None:
            return colors[0], {}

        if cmap is None:
            if self.color_type == 'discrete':
                cmap = numeric_cmap
                scale = max(len(self.color_order) - 1, 1)
            else:
                cmap = categoric_cmap
                scale = 9

        # Map categories to values
        cols = []
        
        # TO DO: what if more than 10 categories?
        self.color_map = {}
        for i, cat in enumerate(self.color_order):
            # TO DO: normalize colours for numeric values!
            self.color_map[cat] = cmap(i/scale)
         
        for value in c:
            if value in self.color_map:
                cols.append(self.color_map[value])
            else:
                cols.append(np.nan)              
            
        return cols, self.color_map


    def get_plot_data(self, data, x, y=None,
                      x_type=None, x_bins=None, x_order=None,
                      color=None, color_order=None):
        """
        Subset our data frame and return processed columns eg binned values if
        needed, subsetted for specific categories
        """
        
        plot_data = pd.DataFrame()
        
        # Remove nans
        cols = [x]
        if y:
            cols.append(y)
            
        if color:
            cols.append(color)

        # Reset index to prevent reordering of columns wrt arrays
        cols = list(set(cols))
        data = data[cols].dropna().reset_index()
        
        if x_type is None or x_type == 'numeric':
            if x_bins is None:
                plot_data['x'] = data[x]
            else:
                plot_data['x'], self.x_bin_width = self.bin_values(data[x].values, bins=x_bins)
        else:
            plot_data['x'] = self.subset_column(data, x, categoric=True, categories=x_order)
        
        if y is not None:
            plot_data['y'] = data[y]
            
        if color is not None:
            plot_data['color'] = self.subset_column(data, color, categoric=True, categories=color_order)

        plot_data = plot_data.dropna()

        return plot_data
    

    def bin_values(self, x, bins=None):
        """
        Assign values of x to binned categories, as in a histogram
        """
        bins = bins if bins else 10
        
        _, bin_edges, bin_num = binned_statistic(x, x, statistic='count', bins=bins)
        
        # Assign each value to the center of its bin
        width = bin_edges[1] - bin_edges[0]
        bins = bin_edges[0] + width * bin_num
        
        return bins, width
    
    
    def format_axis(self, xlabel, ylabel):

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        if self.x_type == 'categoric' or self.x_type == 'text':
            self.ax.set_xticks(range(len(self.x_order)))
            self.ax.set_xticklabels(self.x_order)
            self.ax.grid(axis='x')
            
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(45)
            
        self.make_legend()
        
        self.fig.tight_layout()
        
        
    def category_to_x(self, x, c=None):
        """ 
        Convert category locations to x locations
        
        Arguments
        ---------
        
        x: array of x categories
        
        c: array of color categories
        
        Returns
        -------
        
        xlocs: array
            location of position for the categories on the x axis
        """
        
        if self.x_order is None:
            self.x_order = list(np.unique(x))
        
        x_locs = []
        for i in range(len(x)):
            x_locs.append(self.x_order.index(x[i]))
            
        x_locs = np.array(x_locs)
        
        if c is None:
            return x_locs
        
        width = 0.8 / len(self.color_order)
        c_locs = []
        
        for i in range(len(c)):
            j = self.color_order.index(c[i])
            c_locs.append(width * (j - (len(self.color_order)/2 - 1/2)))
            
        c_locs = np.array(c_locs)
        
        return x_locs + c_locs
                

    def scatterplot(self, jitter=0, alpha=1):

        x, y = self.plot_data['x'].values, self.plot_data['y'].values
        
        if self.color is not None:
            c = self.plot_data['color'].values
        else:
            c = None
        
        if self.x_type != 'numeric':
            # Add color offset
            x = self.category_to_x(x, c=c)
                
        c, _ = self.get_color_mapping(np.array(c))
        x = x + np.random.uniform(-jitter, jitter, len(x)) / 2 / len(self.color_order)
        
        self.ax.scatter(x, y, c=c, alpha=alpha)
        self.format_axis(self.x, self.y)


    def meanplot(self, std=True, style=None):
        """
        Plots mean + sd
        """
        
        if style is None:
            if self.x_type == 'numeric':
                style = 'line'
            else:
                style = 'bar'
        
        if self.color is None:
            color = None
        else:
            color = 'color'

        # Grouped averages
        if self.x_type == 'text':
            stats = text_stats(self.plot_data, 'x', 'y', color=color, stats=['mean'])
            self.x_order = sorted(stats['x'].unique())
        else:
            stats = categorical_stats(self.plot_data, 'x', 'y', color=color, stats=['mean'])

        if color is not None:
            c, cmap = self.get_color_mapping(stats['color'])
        else:
            c, cmap = colors[0], {None: colors[0]}
            

        if style == 'line':
            for i, c_val in enumerate(self.color_order):
                if c_val is None:
                    sub = stats
                else:
                    sub = stats[stats['color'] == c_val]
                    
                if len(stats) == 0:
                    # Some combinations are eliminated from the data so don't
                    # appear when using group by
                    continue
                    
                x, y = sub['x'].values, sub['mean'].values
                if self.x_type != 'numeric':
                    x = self.category_to_x(x)
                    ls, ms = 'none', 12
                else:
                    ls, ms = None, 8
                    
                self.ax.errorbar(x, y,
                                 marker='o', color=cmap[c_val], ms=ms, ls=ls, lw=3)
                
        elif style == 'bar':
            if self.x_type == 'numeric':
                raise ValueError('Bar chart does not work for numeric values')

            x = self.category_to_x(stats['x'].values, c=stats['color'])
            self.ax.bar(x, stats['mean'],
                        width=0.8 / len(self.color_order), color=c,
                        edgecolor=(0.85, 0.85, 0.85), lw=0.75)
        else:
            raise ValueError('Unknown mean plot style')
            
        self.format_axis(self.x, self.y)
        
        return self.plot_data


    def boxplot(self):        
        if self.color is None:
            color = None
        else:
            color = 'color'
            
        # Grouped averages
        stats = categorical_stats(self.plot_data, 'x', 'y', color=color, stats='quantiles')
            
        if color is not None:
            c = stats['color']
        else:
            c = colors[0]
            
        width = 0.8 / len(self.color_order)

        x = self.category_to_x(stats['x'], c=c)[0::5]

        if color is not None:
            c, _ = self.get_color_mapping(stats['color'])
            c = c[0::5]
        else:
            c = None
            
        ten = stats['y'].values[0::5]
        q1 = stats['y'].values[1::5]
        q2 = stats['y'].values[2::5]
        q3 = stats['y'].values[3::5]
        nine = stats['y'].values[4::5]
        
        # Box plots
        self.ax.bar(x, q3 - q1, bottom=q1,
                    width=width, edgecolor='k', lw=1.5, color=c)
        
        # Draw lines
        for i in range(len(x)):
            xs = [x[i] - width / 2, x[i] + width / 2]
            self.ax.plot(xs, [q2[i]] * 2, c='k', lw=1.5)
            
            # Horizontal lines
            xs = [x[i] - width / 4, x[i] + width / 4]
            self.ax.plot(xs, [ten[i]] * 2, c='k', lw=1.5)
            self.ax.plot(xs, [nine[i]] * 2, c='k', lw=1.5)
            
            # Vertical
            self.ax.plot([x[i]]*2, [q3[i], nine[i]], c='k', lw=1.5)
            self.ax.plot([x[i]]*2, [q1[i], ten[i]], c='k', lw=1.5)

        self.format_axis(self.x, self.y)
        
        
    def countplot(self):
        """
        Plots the distribution of x.
        
        Arguments
        ---------
        
        kind: str
            the kind of density to plot. Options are:
                - None/'Count': counts occurences in each bin
                - 'Proportion': gives the proportion of each class in the xbin
                - 'Density': gives the pdf ie the proportion of that xbin for that class
        """

        if self.color is None:
            color = None
        else:
            color = 'color'
            
        # Counts by group and color
        if self.x_type=='text':
            stats = text_stats(self.plot_data, 'x', color=color, max_features=25)
            self.x_order = sorted(stats['x'].unique())
        else:
            stats = categorical_stats(self.plot_data, 'x', color=color)
        
        if self.x_type != 'numeric':
            x = self.category_to_x(stats['x'], c=None)
            width = 0.8
        else:
            x = stats['x']
            width = self.x_bin_width
            
        if self.color is None:
            cmap = {None: colors[0]}
        else:
            _, cmap = self.get_color_mapping(self.plot_data['color'])
        
        bottom = 0

        for i, c_val in enumerate(self.color_order):
            if c_val is None:
                sub = stats
            else:
                sub = stats[stats['color'] == c_val]
                
            if len(sub) == 0:
                # Some combinations are eliminated from the data so don't
                # appear when using group by
                continue
                
            x, y = sub['x'].values, np.nan_to_num(sub['count'].values)
            
            if self.x_type != 'numeric':
                x = self.category_to_x(x)
                
            self.ax.bar(x, y,
                        bottom=bottom, color=cmap[c_val], width=width,
                        edgecolor=(0.85, 0.85, 0.85), lw=0.75)

            bottom += y
            
        self.format_axis(self.x, 'Count')
        
            
def scatterplot(data, x=None, y=None,
                x_type=None, x_bins=None, x_order=None,
                color=None, color_type=None, color_order=None,
                fig=None, ax=None,
                jitter=None, alpha=1):

    p = _RelationalPlotter(data, x, y=y,
                           x_type=x_type, x_bins=x_bins, x_order=x_order,
                           color=color, color_type=color_type, color_order=color_order,
                           fig=fig, ax=ax)
    
    if jitter is None:
        if p.x_type == 'categoric':
            jitter = 0.5
        else:
            jitter = 0
    
    p.scatterplot(jitter=jitter, alpha=1)
    
    
def meanplot(data, x, y,
             x_type=None, x_bins=None, x_order=None,
             color=None, color_type=None, color_order=None,
             fig=None, ax=None):

    p = _RelationalPlotter(data, x, y=y,
                           x_type=x_type, x_bins=x_bins, x_order=x_order,
                           color=color, color_type=color_type, color_order=color_order,
                           fig=fig, ax=ax)

    return p.meanplot()
    
    
def boxplot(data, x, y,
            x_type=None, x_bins=None, x_order=None,
            color=None, color_type=None, color_order=None,
            fig=None, ax=None):
    """
    Create boxplot, split by color. Only for categoric variables
    """
    
    p = _RelationalPlotter(data, x, y=y,
                           x_type='categoric',  x_order=x_order,
                           color=color, color_type=color_type, color_order=color_order,
                           fig=fig, ax=ax)
    
    p.boxplot()
            
    
def countplot(data, x=None, y=None,
              x_type=None, x_bins=None, x_order=None,
              color=None, color_type=None, color_order=None,
              fig=None, ax=None):

    
    if x_bins is None:
        x_bins = 10
    
    p = _RelationalPlotter(data, x, y=None,
                           x_type=x_type, x_bins=x_bins, x_order=x_order,
                           color=color, color_type=color_type, color_order=color_order,
                           fig=fig, ax=ax)

    p.countplot()

    
if __name__ == '__main__':
    titanic = pd.read_csv('D:/Kaggle/Titanic/train.csv')
    houses = pd.read_csv('D:/Kaggle/House Prices/train.csv')
    
#    scatterplot(titanic, 'Pclass', 'Age', x_type='categoric', color='Survived')
#    plt.show()
#
#    scatterplot(titanic, 'Age', 'Fare', color='Sex')
#    plt.show()
#
#    meanplot(houses, 'YearBuilt', 'SalePrice', color='OverallQual', color_order=[1, 3, 5, 7, 9])
#    plt.show()
#
#    meanplot(titanic, 'Sex', 'Age', color='Survived')
#    plt.show()
#    
#    meanplot(titanic, 'Name', 'Age', color='Survived', x_type='text')
#    plt.gcf().set_size_inches(8, 8)
#    plt.show()
#
#    boxplot(houses, 'OverallQual', 'SalePrice')
#    plt.show()
#    
#    boxplot(titanic, 'Sex', 'Age', color='Survived')
#    plt.show()
#
#    countplot(titanic, 'Age', color='Pclass')
#    plt.show()
#
#    countplot(titanic, 'Sex', color='Survived')
#    plt.show()
    
#    countplot(titanic, 'Name', color='Pclass', x_type='text')
#    plt.show()
    
#    meanplot(titanic, 'Name', y='Fare', color=None, x_type='text')
#    plt.show()

#    countplot(houses, '1stFlrSF', color='MSZoning')
#    plt.gcf().set_size_inches(16, 9)

#    countplot(houses, x='1stFlrSF', x_type='categoric', color='2ndFlrSF', color_type='categoric')
#    plt.show()
