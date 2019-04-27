# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:43:02 2019

@author: Mike Staddon

Creates visualisations similar to vega or tableau
"""

import pandas as pd
import numpy as np
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import matplotlib

from colors import categoric_palette, numeric_cmap

marks = ['bar', 'point', 'line', 'heatmap']

from sklearn.feature_extraction.text import CountVectorizer

def aggregate_text(data=None, text=None, max_features=25, order=None, by=None, aggs=None):
    if aggs is None:
        return data
    
    # Put in word order
    vocabulary = None
    if order is not None:
        vocabulary = {order[i]: i for i in range(len(order))}

    counter = CountVectorizer(max_features=max_features, vocabulary=vocabulary)
    counts = counter.fit_transform(data[text]).toarray()

    kvs = counter.vocabulary_
    words = sorted(list(kvs.keys()), key=lambda k: kvs[k])

    # Collection of dataframes to stick together
    stats = []

    # Groupby each word at a time
    for j in range(counts.shape[1]):
        df = data.iloc[counts[:, j] > 0]
        df[text] = words[j]

        stats.append(df.groupby(by=by).agg(aggs).reset_index())

    stats = pd.concat(stats, ignore_index=False)
    
    return stats


class Plotter_():
    def __init__(self, data=None, x=None, y=None, color=None, size=None):
        """
        Processes data and creates plots
        
        Parameters
        ----------
        
        data: DataFrame
            pandas DataFrame containing all data for plotting
            
        x, y: dict, or None
            TO DO
            
        color: dict
            TO DO
            
        size: dict
            TO DO

        fig: mpl Figure
            figure used in plotting
            
        ax: mpl axes
            axes used in plotting
            
        """
        self.raw_data = data

        # Get full options
        self.x = self.clean_parameters(x)
        self.y = self.clean_parameters(y)
        self.color = self.clean_parameters(color)
        self.size = self.clean_parameters(size)
        
        # Bundle up all options so we can loop through them - give better name?
        self.variables = [self.x, self.y, self.color, self.size]
        
        if 'cmap' not in self.color:
            self.color['cmap'] = None
        
        # Clean data for plotting
        self.subset_data()
        self.get_plot_data()
        
        
    def clean_parameters(self, p):
        if p is None:
            p = {}
            
        return self.fill_options(**p)
    
    
    def fill_options(self, column=None, dtype=None, order=None, aggregate=None, bins=None):
        """ Fill in default options of input variables """
        
        v = {}
        v['column']  = column
        
        if dtype is None:
            if self.is_numeric(self.raw_data, column=column):
                dtype = 'numeric'
            else:
                dtype = 'categoric'
        
        v['dtype'] = dtype
        
        
        if v['dtype'] == 'categoric':
            if order is None:
                order = self.get_order(self.raw_data[column])
        
        elif v['dtype'] == 'text':
            order = self.get_words(self.raw_data[column])
                
        v['order'] = order
        v['aggregate'] = aggregate
        v['bins'] = bins
        
        return v
    
    
    def check_variables(self):
        """
        TO DO
        Checks to see if we have valid variables eg no text colours
        """
        pass
        
        
    def is_numeric(self, data, column):
        """ Returns true if a column is numeric """
        if column is None:
            return True
        
        return pd.api.types.is_numeric_dtype(data[column])
    

        
    def get_plot_data(self):
        """ Subset and process the raw data """
        
        # Select column and data we want
        self.subset_data()
        
        # Process the columns
        for v in self.variables:
            self.process_column(**v)

        # Aggregate statistics
        self.aggregate_data()



    def subset_data(self):
        # Subset columns
        cols = set([])

        for v in self.variables:
            if v['column'] is not None:
                cols.add(v['column'])
                    
        cols = list(cols)
                    
        sub = pd.DataFrame()
        sub[cols] = self.raw_data[cols]
                
        # Remove blank values
        self.raw_data = sub.dropna()


    def process_column(self, column=None, dtype=None, values=None, bins=None, **kwargs):
        """
        Process the column by eg binning values or subsetting categories
        """

        if column is None:
            return
        
        # Bin values
        if bins is not None:
            self.raw_data[column] = self.bin_values(self.raw_data[column], bins=bins)
            
        # TO DO: subset values
        
    
    
    def bin_values(self, a, bins=None):
        """
        Assign values of array a to bins, as in a histogram
        """
        bins = bins if bins else 10
        
        _, bin_edges, bin_num = binned_statistic(a, a, statistic='count', bins=bins)
        
        # Assign each value to the center of its bin
        width = bin_edges[1] - bin_edges[0]
        bins = bin_edges[0] + width * (bin_num - 0.5)
        
        return bins


    def get_order(self, a):
        """
        Returns the unique elements of array a in default order
        """

        return sorted(list(a.dropna().unique()))
    
    
    def get_words(self, a, max_features=25):
        """
        Returns the most common words in text column a
        """
        # Get the word count
        counter = CountVectorizer(max_features=max_features)
        counter.fit(a)
    
        # Convret columns into words
        kvs = counter.vocabulary_
        words = sorted(list(kvs.keys()), key=lambda k: kvs[k])
        
        return words


    def aggregate_data(self):
        """
        Aggregate statistics like mean by x, y, or color
        """
        
        by = set([])
        aggs = {col: set([]) for col in self.raw_data.columns}
        
        # Get aggregate variables
        self.count_dummy = None
        aggregate = False
        for v in self.variables:
            if v['aggregate'] is None:
                if v['column'] is not None:
                    by.add(v['column'])
            else:
                # Rename columns to use plot data columns
                aggregate = True
                if v['aggregate'] == 'count':
                    # Use a dummy column for counting
                    self.count_dummy = 'counter'
                    col = 'Count'
                else:
                    aggs[v['column']].add(v['aggregate'])
                    col = v['aggregate'].capitalize() + ' of ' + v['column']
                    
                v['column'] = col
                

        # Convert to lists
        by = list(by)
        for a in aggs:
            aggs[a] = list(aggs[a])

        # If we are not aggregating, then skip this stage
        if not aggregate:
            self.plot_data = self.raw_data
            return

        # If we are grouping by nothing then create a dummy column to use
        self.group_dummy = None
        if len(by) == 0:
            self.group_dummy = 'group'
            by = [self.group_dummy]
            self.raw_data[self.group_dummy] = 0
        
        # Create dummy column for counting
        if self.count_dummy is not None:
            aggs[self.count_dummy] = ['count']
            self.raw_data[self.count_dummy] = 0

        
        # Check if we have text data
        text = None
        if self.x['dtype'] == 'text':
            text = self.x
        elif self.y['dtype'] == 'text':
            text = self.y


        # Get statistics
        if text is None:
            self.plot_data = self.raw_data.groupby(by=by).agg(aggs).reset_index()
        else:
            self.plot_data = aggregate_text(data=self.raw_data,
                                            text=text['column'],
                                            order=text['order'],
                                            by=by,
                                            aggs=aggs)

        # Rename columns to single index form
        cols = by
        for a in aggs:
            for f in aggs[a]:
                if f == 'count':
                    cols += ['Count']
                else:
                    cols += [f.capitalize() + ' of ' + a]

        self.plot_data.columns = cols
        
        if self.group_dummy is not None:
            self.plot_data.drop(columns=[self.group_dummy], inplace=True)

        
    def plot(self, mark, options=None, fig=None, ax=None):

        self.fig = fig
        self.ax = ax
        
        self.make_axis()
        
        if options is None:
            options = {}
        
        if mark == 'point':
            self.scatter(**options)
        elif mark == 'line':
            self.line(**options)
        elif mark == 'bar':
            self.bar(**options)
        elif mark == 'heatmap':
            self.heatmap(**options)
        else:
            raise ValueError('Unknown mark type: {:}'.format(mark))
            
        self.make_legend()
        self.make_labels()
        
        return self.plot_data, self.variables
        
    
    def make_axis(self):
        """ Create figure and axis if needed"""
        if self.fig is None:
            self.fig = plt.gcf()
            self.ax = None
            
        if self.ax is None:
            self.ax = plt.gca()
            
            
    def make_legend(self):
        # TO DO: clean up? Seems like repeated code
        handles = []
        
        if self.color['column'] is not None:
            if self.color['dtype'] == 'categoric':
                
                order = self.color['order']
                colors = self.map_colors(order)
                
                # Title is a invisible line
                h = [matplotlib.lines.Line2D([0], [0], lw=0, 
                           label=str(self.color['column']))]
    
                h += [matplotlib.lines.Line2D([0], [0],
                           marker='o', lw=0,
                           color=colors[i],
                           label=str(cat)) for i, cat in enumerate(order)]
    
                # Blank line
                h += [matplotlib.lines.Line2D([0], [0], lw=0,  label='')]
            
                handles += h
                
            else:
                vmin = self.plot_data[self.color['column']].min()
                vmax = self.plot_data[self.color['column']].max()
            
                # Chooses nicely spaces ticks
                ticker = matplotlib.ticker.MaxNLocator(nbins=4)
                values = ticker.tick_values(vmin, vmax)
                
                cols = self.map_colors(values)
    
                # Title is a blank line
                h = [matplotlib.lines.Line2D([0], [0], lw=0, 
                           label=str(self.color['column']))]
    
                h += [matplotlib.lines.Line2D([0], [0],
                           marker='o', lw=0,
                           color=cols[i],
                           label=str(v)) for i, v in enumerate(values)]
    
                # Blank line
                h += [matplotlib.lines.Line2D([0], [0], lw=0,  label='')]
                
                handles += h

        # Size
        if self.size['column'] is not None:

            vmin = 0
            vmax = self.plot_data[self.size['column']].max()
            
            # Chooses nicely spaces ticks
            ticker = matplotlib.ticker.MaxNLocator(nbins=3)
            values = ticker.tick_values(vmin, vmax)[1:]
            sizes = [v / vmax * 6 for v in values]

            # Title is a blank line
            h = [matplotlib.lines.Line2D([0], [0], lw=0, 
                       label=str(self.size['column']))]

            h += [matplotlib.lines.Line2D([0], [0],
                       marker='o', lw=0, ms=sizes[i],
                       color='k',
                       label=str(v)) for i, v in enumerate(values)]

            # Blank line
            h += [matplotlib.lines.Line2D([0], [0], lw=0,  label='')]
        
            handles += h
                
        # Add in
        self.ax.legend(handles=handles,
                       frameon=False,
                       loc='upper left',
                       bbox_to_anchor=(1, 1))
        

    def make_labels(self):
        """ Label axis and make ticks where needed """
        if self.x['column'] is not None:
            self.ax.set_xlabel(self.x['column'])
            
            if self.x['order'] is not None:
                self.ax.set_xticks(range(len(self.x['order'])))
                self.ax.set_xticklabels(self.x['order'], rotation=90)
        else:
            self.ax.set_xticks([])
            
        if self.y['column'] is not None:
            self.ax.set_ylabel(self.y['column'])
            
            if self.y['order'] is not None:
                self.ax.set_yticks(range(len(self.y['order'])))
                self.ax.set_yticklabels(self.y['order'][::-1])
        else:
            self.ax.set_yticks([])

            
    def map_positions(self, df=None, column=None, dtype=None, order=None,
                      jitter=0, reverse=False, stack=False, width=0.8,
                      **kwargs):

        """ Convert a data column into a plot column 
        
        Arguments
        ---------
        
        TO DO: document

        """
        
        if df is None:
            df = self.plot_data

        # Do we use color?
        if stack or self.color['column'] is None or self.color['dtype'] == 'numeric':
            use_color = False
        else:
            use_color = True
        
        if column is None:
            pos = [0] * len(df)
        else:
            a = df[column].values

            if dtype == 'numeric':
                return a

            # Convert categoric data to positions
            if order is None:
                order = self.get_order(a)
    
            pos = []
            for i in range(len(a)):
                pos.append(order.index(a[i]))
                
            if reverse:
                pos = pos[::-1]


        if use_color:
        
            # Unstack
            c = df[self.color['column']].values
    
            c_order = self.color['order']
            if c_order is None:
                c_order = self.get_order(c)
                
            if len(c_order) == 1:
                return pos
            
            width /= len(c_order)
    
            # Displace by the color position
            for i in  range(len(c)):
                if reverse:
                    pos[i] -= width * ((2 * c_order.index(c[i]) + 1) / 2 - 0.5 * len(c_order))
                else:
                    pos[i] += width * ((2 * c_order.index(c[i]) + 1) / 2 - 0.5 * len(c_order))
                
        if jitter != 0:
            jitter *= width
            pos = np.array(pos) + np.random.uniform(-jitter/2, jitter/2, len(pos))

        return pos
    
    
    def map_colors(self, c=None):
        """
        Maps numbers or categories to colors
        
        Arguments
        ---------
        
        c: array, optional
            if c is given, map c, else map plot_data[color]
        """
        if self.color['column'] is None:
            return categoric_palette[0]
        
        if c is None:
            c = self.plot_data[self.color['column']].values

        if self.color['dtype'] == 'categoric':
            # Map to discrete colors
            order = self.color['order']
            if self.color['cmap'] is None:
                cmap = categoric_palette

            rgb = [cmap[order.index(c[i])] for i in range(len(c))]
            
            return rgb

        else:
            cmin, cmax = c.min(), c.max()
            
            if self.color['cmap'] is None:
                cmap = numeric_cmap
            else:
                cmap = self.color['cmap']
                
            rgb = cmap(np.clip((c - cmin) / (cmax - cmin), 0.0, 1.0))
            
            return rgb
        
        
    def map_sizes(self, s=None, max_size=1):
        """
        Maps numbers or categories to size
        
        Arguments
        ---------
        
        s: array, optional
            if s is given, map s, else map plot_data[size]
        """
        
        if self.size['column'] is None:
            return max_size
        
        if s is None:
            s = self.plot_data[self.size['column']]

        if self.size['dtype'] == 'categoric':
            raise ValueError('Categoric size is not supported!')

        return s / s.max() * max_size

        
            
    def scatter(self, jitterx=0, jittery=0, alpha=0.5, max_size=36):
        # If we aren't aggregating but have text data
        aggregate = False
        for v in self.variables:
            if v['aggregate'] is not None:
                aggregate = True

        # TO DO? Turn into generator?
        # Subset rows by word appearances and plot
        if not aggregate and (self.x['dtype'] == 'text' or self.y['dtype'] == 'text'):
            if self.x['dtype'] == 'text':
                v = self.x
            elif self.y['dtype'] == 'text':
                v = self.y

            counter = CountVectorizer(vocabulary=v['order'])
            counts = counter.fit_transform(self.plot_data[v['column']]).toarray()
            
            # Make plot for each word
            for j in range(len(v['order'])):
                df = self.plot_data.iloc[counts[:, j] > 0]
                df[v['column']] = v['order'][j]
                
                x = self.map_positions(df=df, **self.x, jitter=jitterx, width=0.66)
                y = self.map_positions(df=df, **self.y, jitter=jittery, reverse=True, width=0.66)
                
                if self.color['column'] is None:    
                    c = self.map_colors()
                else:
                    c = self.map_colors(df[self.color['column']].values)
                    
                if self.size['column'] is None:
                    s = self.map_sizes(max_size=max_size)
                else:
                    s = self.map_sizes(s=df[self.size['column']].values, max_size=max_size)

                self.ax.scatter(x, y, color=c, s=s, alpha=alpha)
                
            return


        x = self.map_positions(**self.x, jitter=jitterx, width=0.66)
        y = self.map_positions(**self.y, jitter=jittery, reverse=True, width=0.66)
        c = self.map_colors()        
        s = self.map_sizes(max_size=max_size)

        self.ax.scatter(x, y, color=c, s=s, alpha=alpha)
    
    
    def map_bottom(self, df=None, column=None, by=None, stack=False):
        
        if not stack:
            return 0
        
        if self.color['column'] is None:
            return 0
        
        if self.color['dtype'] == 'numeric':
            return 0
        
        if df is None:
            df = self.plot_data
        
        # Get the bottoms at each position
        bottom = []
        
        for i in range(len(df[column])):
            bot = 0
            
            pos = df[column].iloc[i]
            color = df[self.color['column']].iloc[i]
            
            sub = df[df[column] == pos]
            
            # Count the number of previous of color order
            index = self.color['order'].index(color)
            
            for j in range(index):
                bot += sub[sub[self.color['column']] == self.color['order'][j]][by].sum()
                
            bottom += [bot]
            
        return bottom
    
    
    def bar(self, stack=False, width=1):

        # If x takes real values, map width to bin width
        if self.x['bins'] != None:
            bins = self.plot_data[self.x['column']].unique()
            width *= (max(bins) - min(bins)) / (self.x['bins'] - 1)

        if self.x['dtype'] != 'numeric':
            width *= 0.8

        # Vertical bars
        x = self.map_positions(**self.x, stack=stack, width=width)
        c = self.map_colors()
        
        
        if self.color['dtype'] == 'categoric' and not stack:
            width /= len(self.color['order'])
        
        bottom = self.map_bottom(column=self.x['column'],
                                 stack=stack,
                                 by=self.y['column'])
        self.ax.bar(x,
                    self.plot_data[self.y['column']],
                    width=width,
                    bottom=bottom,
                    color=c)


    def line(self):
        def plot_line(df):
            x = self.map_positions(df, **self.x, width=0.5)
            y = self.map_positions(df, **self.y, reverse=True, width=0.5)
            self.ax.plot(x, y, lw=2.25, marker='o')
        
        
        if self.color['column'] is None:
            plot_line(self.plot_data)
        else:
            for color in self.plot_data[self.color['column']].unique():
                sub = self.plot_data[self.plot_data[self.color['column']] == color]
                sub = sub.sort_values(by=self.x['column'])
                plot_line(sub)
            

    def heatmap(self):
        # TO DO: next version
        pass
        
            
# Generic plot function
def plot(data=None, mark=None, x=None, y=None, color=None, size=None, options=None):
    p = Plotter_(data=data, x=x, y=y, color=color, size=size)
    return p.plot(mark=mark, options=options)



if __name__ == '__main__':
    titanic = pd.read_csv('D:/Kaggle/Titanic/train.csv')
    houses = pd.read_csv('D:/Kaggle/House Prices/train.csv')
    
    count = 1
    
    ### Bar plots
    plot(data=titanic,
         mark='bar',
         x={'column': 'Sex'},
         y={'column': 'Parch', 'aggregate': 'mean'},
         color={'column': 'Pclass', 'dtype': 'categoric'})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    
    plot(data=titanic,
         mark='bar',
         x={'column': 'Sex'},
         y={'aggregate': 'count'},
         color={'column': 'Pclass', 'dtype': 'categoric'})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    
    plot(data=titanic,
         mark='bar',
         x={'column': 'Sex'},
         y={'aggregate': 'count'},
         color={'column': 'Pclass', 'dtype': 'categoric'},
         options={'stack': True})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='bar',
         x={'column': 'Age', 'bins': 10},
         y={'column': 'Fare', 'aggregate': 'sum'},
         color={'column': 'Pclass', 'dtype': 'categoric'},
         options={'stack': True})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='bar',
         x={'column': 'Name', 'dtype': 'text'},
         y={'aggregate': 'count'})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='bar',
         x={'column': 'Name', 'dtype': 'text'},
         y={'aggregate': 'count'},
         color={'column': 'Survived', 'dtype': 'categoric'},
         options={'stack': True})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()

    plot(data=houses,
         mark='bar',
         x={'column': 'OverallQual'},
         y={'aggregate': 'count'},
         color={'column': 'SalePrice', 'aggregate': 'mean'})

    ## Line plots
    plot(data=titanic,
         mark='line',
         x={'column': 'Sex'},
         y={'column': 'Parch', 'aggregate': 'mean'},
         color={'column': 'Pclass', 'dtype': 'categoric'})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='line',
         x={'column': 'Sex'},
         y={'column': 'Parch', 'aggregate': 'mean'},
         color={'column': 'Sex', 'dtype': 'categoric'})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()

    plot(data=houses,
         mark='line',
         x={'column': 'YearBuilt', 'bins': 10},
         y={'column': 'SalePrice', 'aggregate': 'mean'},
         color={'column': 'OverallQual', 'dtype': 'categoric'})

    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()

    ## Point plots
    plot(data=titanic,
         mark='point',
         x={'column': 'SibSp', 'aggregate': 'mean'},
         y={'column': 'Parch'},
         size={'aggregate': 'count', 'column': 'Pclass'},
         color={'aggregate': 'mean', 'column': 'SibSp'})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='point',
         x={'column': 'SibSp', 'aggregate': 'mean'})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='point',
         x={'column': 'Sex'},
         y={'aggregate': 'count'})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()

    plot(data=titanic,
         mark='point',
         y={'column': 'Name', 'dtype': 'text'},
         x={'column': 'Fare'},
         color={'column': 'Pclass', 'dtype': 'categoric'})
    
    plt.gcf().set_size_inches(5, 12.5)
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()

    plot(data=titanic,
         mark='point',
         y={'column': 'Sex'},
         x={'column': 'Fare'},
         color={'column': 'Pclass', 'dtype': 'categoric'},
         options={'jittery': 0.5, 'alpha': 0.33})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()
    
    plot(data=titanic,
         mark='point',
         y={'column': 'Sex'},
         x={'column': 'Fare', 'bins': 50},
         color={'column': 'Pclass', 'dtype': 'categoric'},
         size={'aggregate': 'count'},
         options={'jittery': 0, 'alpha': 0.8, 'max_size': 360})
    
    plt.gcf().savefig('images/fig_{:}.png'.format(count), bbox_inches='tight')
    count += 1
    plt.show()