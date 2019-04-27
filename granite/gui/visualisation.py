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

from gui.widgets import OptionsBox, MplFigureCanvas, AddRemoveButton, ItemsView

from matplotlib.figure import Figure


class NumericFeatureBox(QtWidgets.QFrame):
    # Contains numeric features for pair plots
    def __init__(self, project, varOptions, parent=None, feature=None):
        QtWidgets.QFrame.__init__(self, parent=parent)
        
        self.project = project
        self.varOptions = varOptions
        self.parent=parent
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        removeBut = AddRemoveButton('-', add=False)
        removeBut.clicked.connect(self.RemoveFeature)
        self.grid.addWidget(removeBut, 0, 0)
        
        self.featureBox = OptionsBox()
        self.featureBox.addItems(varOptions)
        self.featureBox.currentIndexChanged[str].connect(self.ChangeFeature)
        self.grid.addWidget(self.featureBox, 0, 1)
    
        self.grid.setColumnStretch(1, 1)
        
        if feature is None:
            feature = varOptions[0]
            
        self.ChangeFeature(feature)
            
            
    def ChangeFeature(self, feature):
        self.feature = feature
        
        
    def GetOptions(self):
        return self.feature
    
    
    def RemoveFeature(self):
        self.parent.RemoveFeature(self)
        
        
# Contains categoric features for x axis in factor plots
class CategoricFeatureBox(QtWidgets.QFrame):
    def __init__(self, project, varOptions, parent=None,
                 feature=None, dtype=None, **kwargs):
        
        QtWidgets.QFrame.__init__(self, parent=parent)
        
        self.project = project
        self.varOptions = varOptions
        self.parent=parent
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        removeBut = AddRemoveButton('-', add=False)
        removeBut.clicked.connect(self.RemoveFeature)
        self.grid.addWidget(removeBut, 0, 0)
        
        self.featureBox = OptionsBox()
        self.featureBox.addItems(varOptions)
        self.featureBox.currentIndexChanged[str].connect(self.ChangeFeature)
        self.grid.addWidget(self.featureBox, 0, 1)
        
        self.grid.setColumnStretch(1, 1)

        self.featureType = OptionsBox()
        self.grid.addWidget(self.featureType, 1, 0, 1, 2)
        self.featureType.hide()
        
        if feature is None:
            feature = varOptions[0]
            
        self.ChangeFeature(feature)
        
        if dtype is not None:
            fType = {'categoric': 'Categoric',
                     'numeric': 'Numeric (binned)'}[dtype]
            index = self.featureType.findText(fType)
            self.featureType.setCurrentIndex(index)
            
            
    def ChangeFeature(self, feature):
        self.feature = feature

        items = ['Categoric']
        if feature in self.project.get_numeric_columns():
            items = ['Numeric (binned)'] + items
            
        self.featureType.clear()
        self.featureType.addItems(items)
        self.featureType.show()
        
        
    def GetOptions(self):
        opts = {}
        if self.featureType.currentText() == 'Categoric':
            opts['dtype'] = 'categoric'
            opts['categories'] = None
        else:
            opts['dtype'] = 'numeric'
            opts['bins'] = None
        
        return self.feature, opts
    
    
    def RemoveFeature(self):
        self.parent.RemoveFeature(self)
        

# Contains feature and style to plot on y axis for factor plots
class PlotStyleBox(QtWidgets.QFrame):
    def __init__(self, project, varOptions, parent=None,
                 style=None, feature=None):
        QtWidgets.QFrame.__init__(self, parent=parent)
        
        self.project = project
        self.varOptions = list(varOptions)
        self.parent=parent
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        removeBut = AddRemoveButton('-', add=False)
        removeBut.clicked.connect(self.RemoveFeature)
        self.grid.addWidget(removeBut, 0, 0)
        
        # Plot style
        self.styleBox = OptionsBox()
        
        styleOptions = ['Frequency', 'Total', 'Mean', 'Box', 'Jitter']
        self.styleBox.addItems(styleOptions)
        self.styleBox.currentIndexChanged[str].connect(self.ChangePlotStyle)
        self.grid.addWidget(self.styleBox, 0, 1)
        
        self.featureBox = OptionsBox()
        self.featureBox.addItems(self.varOptions)
        self.featureBox.currentIndexChanged[str].connect(self.ChangeFeature)
        self.grid.addWidget(self.featureBox, 1, 0, 1, 2)
        self.featureBox.hide()
    
        self.grid.setColumnStretch(1, 1)

        self.ChangePlotStyle('Frequency')
        self.ChangeFeature(varOptions[0])
            
        if style is not None:
            style = style.title()
            self.styleBox.setCurrentIndex(styleOptions.index(style))
            
            if style != 'Frequency':
                self.featureBox.setCurrentIndex(self.varOptions.index(feature))

            
    def ChangePlotStyle(self, style):
        self.style = style

        if style == 'Frequency':
            self.featureBox.hide()
        else:
            self.featureBox.show()
        
        
    def ChangeFeature(self, feature):
        self.feature = feature
        
        
    def GetOptions(self):
        if self.style == 'Frequency':
            return 'frequency', None
        else:
            return self.style.lower(), self.feature
    
    
    def RemoveFeature(self):
        self.parent.RemoveFeature(self)
        
      
# Contains colour options for both pair and factor plots
class ColorBox(QtWidgets.QFrame):
    def __init__(self, project, parent=None,
                 color=None, color_categoric=False):
        
        QtWidgets.QFrame.__init__(self, parent=parent)
        
        self.project = project
        self.parent=parent
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        colorOptions = [''] + project.get_all_columns()

        self.featureBox = OptionsBox()
        self.featureBox.addItems(colorOptions)
        self.featureBox.currentIndexChanged[str].connect(self.ChangeFeature)
        self.grid.addWidget(self.featureBox, 0, 0)
        
        self.colorType = OptionsBox()
        self.colorType.addItems(['Numeric', 'Categoric'])
        self.grid.addWidget(self.colorType, 1, 0)
        self.colorType.hide()
        
        self.feature = None
        self.categoric = False
        
        if color is not None:
            self.featureBox.setCurrentIndex(colorOptions.index(color))
            self.ChangeFeature(color)
            self.colorType.setCurrentIndex(1*color_categoric)
        
        
    def ChangeFeature(self, feature):
        self.feature = feature
        if feature == '':
            self.feature = None
            self.colorType.hide()
        else:
            items = ['Categoric']
            if feature in self.project.get_numeric_columns():
                items = ['Numeric'] + items
                
            self.colorType.clear()
            self.colorType.addItems(items)
            self.colorType.show()
                
        
    def GetOptions(self):
        color_categoric = self.colorType.currentText() == 'Categoric'
        return self.feature, color_categoric
    

class FeaturesList(QtWidgets.QFrame):
    def __init__(self, project, parent=None, varType=None,
                 features=None, options=None):
        QtWidgets.QFrame.__init__(self, parent=parent)
        
        self.project = project
        self.varType = varType
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        # Only allow numeric columns
        if varType == 'categoric':
            self.columns = self.project.get_all_columns()
        else:
            self.columns = self.project.get_numeric_columns()
        
        self.featureBoxes = []

        # Initially only an add widget option
        self.MakeAddOption()
        
        # Fill in preset features
        if features is not None:
            if options is None:
                options = [None]*len(features)
                
            # y options aren't a dictionary, maybe it should be
            if varType == 'plotstyle':
                options = [{'style': opt} for opt in options]
            elif varType == 'categoric':
                options = [options[opt] for opt in options]
                
            for feature, opts in zip(features, options):
                self.AddFeature(feature=feature, options=opts)
            

    def MakeAddOption(self):
        w = AddRemoveButton('+', add=True)
        w.clicked.connect(self.AddFeature)
        self.featureBoxes += [w]
        self.grid.addWidget(w, len(self.featureBoxes), 0)
        
    
    def AddFeature(self, feature=None, options=None):

        if options is None:
            options = {}
        
        if self.varType == 'numeric':
            fb = NumericFeatureBox(self.project, self.columns, parent=self,
                                   feature=feature, **options)
            
        elif self.varType == 'categoric':
            fb = CategoricFeatureBox(self.project, self.columns, parent=self,
                                     feature=feature, **options)
            
        elif self.varType == 'plotstyle':
            fb = PlotStyleBox(self.project, self.columns, parent=self,
                                feature=feature, **options)
            
        self.featureBoxes[-1].setParent(None)
        self.featureBoxes[-1] = fb
        self.grid.addWidget(fb, len(self.featureBoxes), 0)
        self.MakeAddOption()
        
        
    def RemoveFeature(self, widget):
        self.featureBoxes.remove(widget)
        widget.setParent(None)
        
        # Replot all feaures
        for i, fb in  enumerate(self.featureBoxes):
            self.grid.addWidget(fb, i, 0)
            
            
    def GetOptions(self):
        if self.varType == 'numeric':
            return [fb.GetOptions() for fb in self.featureBoxes[:-1]]
        
        elif self.varType == 'categoric':
            xs = []
            xs_options = {}
            
            for fb in self.featureBoxes[:-1]:
                feature, opts = fb.GetOptions()
                xs += [feature]
                xs_options[feature] = opts
                
            return xs, xs_options
        
        elif self.varType == 'plotstyle':
            ys = []
            styles = []
            
            for fb in self.featureBoxes[:-1]:
                style, y = fb.GetOptions()
                ys += [y]
                styles += [style]
                
            return ys, styles


    def SetOptions(self, options):
        # Set widget options based on saved values
        pass

    
class PairPlotOptions(QtWidgets.QWidget):
    def __init__(self, project, fig, canvas,
                 xs=None, color=None, color_categoric=False):
        
        super(PairPlotOptions, self).__init__()
        
        self.project = project
        self.fig = fig
        self.canvas = canvas
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        self.grid.addWidget(QtWidgets.QLabel('Features'), 0, 0)
        
        self.xOptions = FeaturesList(self.project, parent=self, varType='numeric',
                                     features=xs)
        
        self.grid.addWidget(self.xOptions, 1, 0)
        
        self.grid.addWidget(QtWidgets.QLabel('Color'), 2, 0)
        
        self.colorOptions = ColorBox(self.project, parent=self,
                                     color=color, color_categoric=color_categoric)
        
        self.grid.addWidget(self.colorOptions, 3, 0)

        self.grid.setRowStretch(4, 1)
        

    def GetOptions(self):
        xs = self.xOptions.GetOptions()
        color, color_categoric = self.colorOptions.GetOptions()
        
        return {'xs': xs,
                'color': color,
                'color_categoric': color_categoric}
        
        
class FactorPlotOptions(QtWidgets.QWidget):
    def __init__(self, project, fig, canvas, options=None,
                 xs=None, xs_options=None, ys=None, styles=None,
                 color=None, color_categoric=None):
        super(FactorPlotOptions, self).__init__()

        # Load dataset to test
        self.project = project
        self.fig = fig
        self.canvas = canvas

        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        # x values
        self.grid.addWidget(QtWidgets.QLabel('Columns'), 0, 0)
        self.xOptions = FeaturesList(self.project, parent=self, varType='categoric',
                                     features=xs, options=xs_options)
        
        self.grid.addWidget(self.xOptions, 1, 0)


        # y values
        self.grid.addWidget(QtWidgets.QLabel('Rows'), 2, 0)
        self.yOptions = FeaturesList(self.project, parent=self, varType='plotstyle',
                                     features=ys, options=styles)
        
        self.grid.addWidget(self.yOptions, 3, 0)

        # color
        self.grid.addWidget(QtWidgets.QLabel('Color'), 4, 0)
        self.colorOptions = ColorBox(self.project, parent=self,
                                     color=color, color_categoric=color_categoric)
        self.grid.addWidget(self.colorOptions, 5, 0)

        self.grid.setRowStretch(6, 1)
        
    
    def GetOptions(self):
        xs, xs_options = self.xOptions.GetOptions()
        ys, styles = self.yOptions.GetOptions()
        color, color_categoric = self.colorOptions.GetOptions()
        
        return {'xs': xs,
                'xs_options': xs_options,
                'ys': ys,
                'styles': styles,
                'color': color,
                'color_categoric': color_categoric}
        
        
class PlottingFrame(QtWidgets.QFrame):
    def __init__(self, visFrame, project, style=None, options=None):
        super(PlottingFrame, self).__init__()
        
        self.visFrame = visFrame
        self.project = project
        self.style=None
        
        if style is None:
            style = 'Pair Plot'
        
        if options is None:
            self.plotOptions = {}
        else:
            self.plotOptions = options
        
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        # Savebar
        saveBar = QtWidgets.QFrame()
        grid = QtWidgets.QGridLayout()
        saveBar.setLayout(grid)
        
        but = QtWidgets.QPushButton('Back')
        but.clicked.connect(self.Back)
        grid.addWidget(but, 0, 0)
        
        but = QtWidgets.QPushButton('Save')
        but.clicked.connect(self.Save)
        grid.addWidget(but, 0, 1)
        
        grid.setColumnStretch(2, 1)        
        
        self.grid.addWidget(saveBar, 0, 0, 1, 2)
        
        
        # Plot canvas
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.canvas = MplFigureCanvas(self.fig)
        self.grid.addWidget(self.canvas, 1, 1)
        
        # Plotting options
        sideBar = QtWidgets.QFrame()
        self.sideGrid = QtWidgets.QGridLayout()
        sideBar.setLayout(self.sideGrid)
        
        self.sideGrid.addWidget(QtWidgets.QLabel('Style'), 0, 0)
        self.styleOptions = OptionsBox()
        self.styleOptions.addItems(['Pair Plot', 'Factor Plot'])
        self.styleOptions.currentIndexChanged[str].connect(self.ChangeStyle)
        self.sideGrid.addWidget(self.styleOptions, 1, 0)

        # Blanked widget - replaced next line
        self.optionsFrame = QtWidgets.QWidget()
        self.ChangeStyle(style, options=options)
        
        self.sideGrid.addWidget(self.optionsFrame, 2, 0)
        
        but = QtWidgets.QPushButton('Plot')
        but.clicked.connect(self.Plot)
        self.sideGrid.addWidget(but, 3, 0)
        
        self.sideGrid.setRowStretch(2, 1)
        
        self.grid.addWidget(sideBar, 1, 0)
        
        # Stretch plot row and column
        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        
        if options is not None:
            self.Plot()
        

    def ChangeStyle(self, style, options=None):
        if style == self.style:
            return
        
        self.style = style
        
        if options is None:
            options = {}
        
        # Replace options frame
        self.optionsFrame.setParent(None)
        if style == 'Pair Plot':
            self.optionsFrame = PairPlotOptions(self.project,
                                                self.fig, self.canvas,
                                                **options)
        else:
            self.optionsFrame = FactorPlotOptions(self.project,
                                                  self.fig, self.canvas,
                                                  **options)
            
        self.sideGrid.addWidget(self.optionsFrame, 2, 0)
        
    
    def Plot(self):
        self.plotOptions = self.optionsFrame.GetOptions()
        self.project.plot(self.style, self.fig, **self.plotOptions)
        
        self.canvas.draw()

    
    def Back(self):
        # TO DO: Do you want to save!?
        self.visFrame.ClosePlotFrame()
    
    
    def Save(self):
        self.plotOptions = self.optionsFrame.GetOptions()
        self.visFrame.SavePlot(self.style, self.plotOptions)

        
class VisualisationTab(QtWidgets.QFrame):
    """ Frame containing saved and new visualisations
    """
    def __init__(self, project, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        
        """
        Load previous project plots
        """
        
        self.project = project
        
        self.currentPlotIndex = None
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        
        self.savedPlots = []
        
        self.itemsFrame = ItemsView(self)
        self.grid.addWidget(self.itemsFrame, 0, 0)


    def NewItem(self):
        # Make a new plot
        self.itemsFrame.hide()
        self.plotFrame = PlottingFrame(self, self.project)
        self.grid.addWidget(self.plotFrame)
        
        self.currentPlotIndex = None

    
    def SavePlot(self, style, options):
        self.savedPlots.append([style, options])

        # Add item to list
        text = 'Style: '+style
        
        if 'xs' in options:
            text += '\nX: '+str(options['xs'])
            
        if 'ys' in options:
            text += '\nY: '+str(options['ys'])
            
        if 'color' in options:
            text += '\nColor: '+str(options['color'])
            
        widget = QtWidgets.QPushButton(text)
        
        self.itemsFrame.AddItem(widget, self.currentPlotIndex)


    def SelectItem(self, index):
        # Load a plot
        item = self.savedPlots[index]
        
        self.itemsFrame.hide()
        self.plotFrame = PlottingFrame(self, self.project,
                                       style=item[0], options=item[1])
        self.grid.addWidget(self.plotFrame)
        
        self.currentPlotIndex = index


    def ClosePlotFrame(self):
        self.plotFrame.setParent(None)
        self.itemsFrame.show()