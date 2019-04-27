# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:00:00 2018

@author: Mike Staddon
"""

import pandas as pd
import numpy as np
import os

os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets

from .widgets import (OptionsBox, MplFigureCanvas, ItemsView, StyleLine,
                      ValueSlider, WindowManager, BoxFrame)
                      
from ..plotting.relational import scatterplot, boxplot, countplot, meanplot

from matplotlib.figure import Figure


class VariableOptions(QtWidgets.QFrame):
    def __init__(self, project, variable_name,
                 numeric=False, binned=False, categoric=False, text=False):
        """
        Create an options box with the specific type options
        """
        
        QtWidgets.QFrame.__init__(self)
        self.setObjectName('background')
        
        self.project = project
        
        self.variable_name = variable_name
        
        # Create ui
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setRowStretch(100, 1)
        
        self.variable = None
        self.dtype = None
        
        self.MakeWidgets()
        self.UpdateTypes(numeric=numeric, binned=binned, categoric=categoric, text=text)

        
    def MakeWidgets(self):
        # Separates the ui elements
        self.grid.addWidget(StyleLine())
        
        self.grid.addWidget(QtWidgets.QLabel(self.variable_name))
        
        # Make all elements and hide when appropriate
        
        # Variable
        self.variableBox = OptionsBox()
        self.variableBox.currentIndexChanged[str].connect(self.ChangeVariable)
        self.grid.addWidget(self.variableBox)
        
        # Variable type
        self.typeBox = OptionsBox()
        self.typeBox.currentIndexChanged[str].connect(self.ChangeType)
        self.grid.addWidget(self.typeBox)
        
        # Numeric bins options
        self.binsSlider = ValueSlider([5, 50], default=10)
        self.grid.addWidget(self.binsSlider)
        
        # Warning
        self.warningLabel = QtWidgets.QLabel('')
        self.warningLabel.setObjectName('warning')
        self.grid.addWidget(self.warningLabel)
        

    def UpdateTypes(self, numeric=False, binned=False, categoric=False, text=False):
        self.numeric = numeric
        self.binned = binned
        self.categoric = categoric
        self.text = text
        
        # Add variable options
        if self.categoric or self.text:
            xs = sorted(self.project.get_all_columns())
        else:
            xs = sorted(self.project.get_numeric_columns())
        
        # Store old variables - the index is changed when adding items
        oldvar = self.variable
        oldtype = self.dtype
        
        self.variableBox.clear()
        self.variableBox.addItems(xs)
        
        self.variable = oldvar
        
        # Set to default
        if len(xs) > 0:
            self.ChangeVariable(xs[0])
        else:
            self.variable = None
            
        # Try old types
        if oldvar is not None and oldtype is not None:
            # Convert string to plotting dtype format
            self.SetOptions(oldvar, dtype=oldtype.split(' ')[0].lower())
        
        
    def ChangeVariable(self, variable):
        if variable == '':
            variable = None
            
        self.variable = variable
        
        numeric = variable in self.project.get_numeric_columns()

        dtypes = []
        if self.numeric and numeric:
            dtypes.append('Numeric')
        if self.binned and numeric:
            dtypes.append('Numeric (binned)')
        if self.categoric:
            dtypes.append('Categoric')
        if self.text and not numeric:
            dtypes.append('Text')
            
        self.typeBox.clear()
        self.typeBox.addItems(dtypes)
        
        if len(dtypes) > 0:
            self.ChangeType(dtypes[0])
        
        
    def ChangeType(self, dtype):
        self.dtype = dtype
        
        if dtype == 'Numeric (binned)':
            self.binsSlider.show()
        else:
            self.binsSlider.hide()
            
        # Change values
        warning = ''
        
        if self.dtype == 'Categoric' and self.variable is not None:
            if self.project.data[self.variable].nunique() > 25:
                warning = 'Showing top 25 values'
        
        self.warningLabel.setText(warning)


    def GetOptions(self):
        variable = self.variable
        
        if self.variable is None:
            return None
        
        if self.dtype == 'Numeric (binned)':
            dtype, bins = 'numeric', self.binsSlider.value()
        else:
            dtype, bins = self.dtype.lower(), None
            
        return variable, dtype, bins
    
    
    def SetOptions(self, variable, dtype=None, bins=None):
        # Change the indices of the optionboxes
        index = self.variableBox.findText(variable)
        if index >= 0:
            self.variableBox.setCurrentIndex(index)

        if dtype is not None:          
            if dtype == 'numeric':
                if bins == None:
                    dtype = 'Numeric'
                else:
                    dtype = 'Numeric (binned)'
            elif dtype == 'categoric':
                dtype = 'Categoric'
            elif dtype == 'text':
                dtype = 'Text'

            index = self.typeBox.findText(dtype)
            if index >= 0:
                self.typeBox.setCurrentIndex(index)

        if bins is not None:
            self.binsSlider.SetValue(bins)
            
    
class ColorOptions(QtWidgets.QFrame):
    def __init__(self, project):
        """
        Create an options box with the specific type options
        """
        
        QtWidgets.QFrame.__init__(self)
        self.setObjectName('background')
        
        self.project = project
        
        # Create ui
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setRowStretch(100, 1)
        
        self.MakeWidgets()
        
        
    def MakeWidgets(self):
        # Separates the ui elements
        self.grid.addWidget(StyleLine())
        
        self.grid.addWidget(QtWidgets.QLabel('Color'))
        
        # Make all elements and hide when appropriate
        
        # Variable
        self.variableBox = OptionsBox()
        xs = [''] + sorted(self.project.get_all_columns())

        self.variableBox.addItems(xs)
        self.variableBox.currentIndexChanged[str].connect(self.ChangeVariable)
        self.grid.addWidget(self.variableBox)
        
        
        # Variable type
        self.typeBox = OptionsBox()
        self.typeBox.currentIndexChanged[str].connect(self.ChangeType)
        self.grid.addWidget(self.typeBox)
        
        
        # Warning
        self.warningLabel = QtWidgets.QLabel('')
        self.warningLabel.setObjectName('warning')
        self.grid.addWidget(self.warningLabel)
        
        
        # Change to default
        self.ChangeVariable(xs[0])
        
        
    def ChangeVariable(self, variable):
        if variable == '':
            variable = None
            
        self.variable = variable
        
        # Update type
        if variable is None:
            types = []
        elif variable in self.project.get_numeric_columns():
            types = ['Discrete', 'Categoric']
        else:
            types = ['Categoric']
            
        self.typeBox.clear()
        self.typeBox.addItems(types)
        
        if len(types) > 0:
            self.ChangeType(types[0])
        else:
            self.ChangeType(None)
        
        
    def ChangeType(self, dtype):
        if type(dtype) is str:
            dtype = dtype.lower()

        self.variableType = dtype
        
        # Change values
        warning = ''
        if self.variable is not None:
            if self.project.data[self.variable].nunique() > 10:
                warning = 'Showing top 10 values'
        
        self.warningLabel.setText(warning)
        

    def GetOptions(self):
        return self.variable, self.variableType
    
    
    def SetOptions(self, color, color_type):
        if color is None:
            color = ''
            
        index = self.variableBox.findText(color)
        if index >= 0:
            self.variableBox.setCurrentIndex(index)


class PlotOptions(QtWidgets.QFrame):
    def __init__(self, project, plotFrame, style=None, options=None):
        """
        Contains the plot style and updates options for that style
        """
        super(PlotOptions, self).__init__()
        
        self.project = project
        self.plotFrame = plotFrame

        self.setObjectName('background')
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setRowStretch(5, 1)
        self.setContentsMargins(0,0,0,0)
        
        # Style
        l = QtWidgets.QLabel('Style')
        self.grid.addWidget(l, 0, 0)
        
        style_options = ['Count', 'Scatter', 'Mean', 'Box Plot']

        self.styleBox = OptionsBox()
        self.styleBox.addItems(style_options)
        self.styleBox.currentIndexChanged[str].connect(self.ChangeStyle)
        self.grid.addWidget(self.styleBox, 1, 0)
        
        # Placeholder widgets
        self.xOptions = VariableOptions(self.project, 'x')
        self.grid.addWidget(self.xOptions, 2, 0)
        
        self.yOptions = VariableOptions(self.project, 'y', numeric=True)
        self.grid.addWidget(self.yOptions, 3, 0)
        
        self.colorOptions = ColorOptions(self.project)
        self.grid.addWidget(self.colorOptions, 4, 0)
        
        # Plot
        but = QtWidgets.QPushButton('Plot')
        but.clicked.connect(self.Plot)
        self.grid.addWidget(but, 6, 0)
        
        # Add in the style specific options
        if style is None:
            self.ChangeStyle(style_options[0])
        else:
            self.ChangeStyle(style)
            
            # Put in options
            self.xOptions.SetOptions(options['x'], dtype=options['x_type'], bins=options['x_bins'])
            self.yOptions.SetOptions(options['y'])
            self.colorOptions.SetOptions(options['color'], options['color_type'])
            
            self.Plot()
        
        
    def ChangeStyle(self, style):
        self.style = style

        # Update variable options
        if style == 'Count':
            numeric, binned, categoric, text = False, True, True, True
        elif style == 'Scatter':
            numeric, binned, categoric, text = True, False, True, False
        elif style == 'Mean':
            numeric, binned, categoric, text = True, True, True, True
        elif style == 'Box Plot':
            numeric, binned, categoric, text = False, False, True, False

        self.xOptions.UpdateTypes(numeric=numeric,
                                  binned=binned,
                                  categoric=categoric,
                                  text=text)
        
        if style != 'Count':
            self.yOptions.show()
        else:
            self.yOptions.hide()
        
        
    def GetOptions(self):
        style = self.style
        
        options = {}
        
        opts = self.xOptions.GetOptions()
        if opts is None:
            return
        else:
            options['x'], options['x_type'], options['x_bins'] = opts
        
        if style is not 'Count':
            options['y'], _, _ = self.yOptions.GetOptions()
        else:
            options['y'] = None
            
        options['color'], options['color_type'] = self.colorOptions.GetOptions()
        
        return style, options
        

    def Plot(self):
        self.plotFrame.Plot(*self.GetOptions())
    
        

class PlottingFrame(QtWidgets.QFrame):
    def __init__(self, visFrame, project, index, style=None, options=None):
        """
        Window containing the plot and it's options
        
        index: int
            index of the plot in the project
        """
        super(PlottingFrame, self).__init__()

        self.visFrame = visFrame
        self.project = project
        self.index = index
        self.style=None
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        
        # Plot canvas
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = MplFigureCanvas(self.fig)
        self.grid.addWidget(self.canvas, 1, 1)
        
        # Plotting options
        self.optionsFrame = PlotOptions(self.project, self, style=style, options=options)
        self.grid.addWidget(self.optionsFrame, 1, 0)
        
        # Stretch plot row and column
        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        
    
    def Plot(self, style, plotOptions):
        self.style, self.plotOptions = style, plotOptions
        
        if self.plotOptions is None:
            return
        
        self.ax.cla()
        
        plotters = {'Count': countplot,
                    'Mean': meanplot,
                    'Box Plot': boxplot,
                    'Scatter': scatterplot}
        
        plotter = plotters[self.style]
        plotter(self.project.data, fig=self.fig, ax=self.ax, **self.plotOptions)
        
        self.canvas.draw()
        
        # Also save options
        self.project.save_plot(self.style, self.plotOptions, self.index)
        self.visFrame.AddItem(self.style, self.plotOptions, self.index)
        
        
class PlotIcon(BoxFrame):
    """ The icon to select an experiment with """
    def __init__(self, style, options):
        BoxFrame.__init__(self, style)
        
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

        row = 0
        for i, o in enumerate(options):
            # Only show non-default options
            if options[o] is None:
                continue
            
            # Clean up strings
            text = o.replace('_', ' ')
                
            self.grid.addWidget(QtWidgets.QLabel(text), row, 0)
            
            # Clean up strings
            if o == 'x_type':
                if options['x_bins'] != None:
                    text = 'Numeric (binned)'
                else:
                    text = options[o].capitalize()
            elif o == 'color_type':
                text = options[o].capitalize()
            else:
                text = options[o]
            
            label = QtWidgets.QLabel(str(text))
            self.grid.addWidget(label, row, 1)
            
            row += 1
            
        
        f = QtWidgets.QFrame()
        self.grid.addWidget(f, 0, 2, len(options), 1)
        
        grid = QtWidgets.QGridLayout(f)
        grid.setContentsMargins(0,0,0,0)

        self.button = QtWidgets.QPushButton('Open')
        grid.addWidget(self.button, 0, 0)
        
        self.removeButton = QtWidgets.QPushButton('Delete')
        self.removeButton.setObjectName('subtle')
        grid.addWidget(self.removeButton, 1, 0)

        
class VisualisationTab(WindowManager):
    def __init__(self, project):
        """
        Frame containing saved and new visualisations
        """
        self.project = project
        self.itemsFrame = ItemsView(self, has_button=True, has_remove=True)
        self.itemsFrame.setMaximumWidth(600)
        
        # TO DO: load and save items
        for style, options in self.project.plots:
            # Add to the items frame
            self.AddItem(style, options)

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setObjectName('background')
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.itemsFrame)
        self.scrollArea.setMaximumWidth(600)

        WindowManager.__init__(self, self.scrollArea, 'Visualisations')
        

    def NewItem(self):
        self.OpenWindow(PlottingFrame(self, self.project, len(self.project.plots)), '')
        
        
    def AddItem(self, style, options, index=None):
        widget = PlotIcon(style, options)
        self.itemsFrame.AddItem(widget, index=index)
        
        
    def RemoveItem(self, index):
        # Remove from project
        self.project.remove_plot(index)


    def SelectItem(self, index):
        style, options = self.project.plots[index]
        self.OpenWindow(PlottingFrame(self, self.project, index, style=style, options=options), '')
