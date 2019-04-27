# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:00:00 2018

@author: Mike Staddon
"""

from matplotlib.figure import Figure

import pandas as pd
import numpy as np
import os

os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets, QtCore, QtGui

from gui.style import max_width, app_css

from gui.widgets import (OptionsBox, MplFigureCanvas, AddRemoveButton,
                     ItemsView, ValueSlider, TableNumericItem,
                     LabelBox, FormatNumber, ValueBar, PercentBar,
                     make_values_table, PagedTable, NumericEdit,
                     NumericLineEdit, WindowManager, FormatScore,
                     StyleLine, BoxFrame, FormatTime, CenterWidget)


from sklearn.pipeline import Pipeline, FeatureUnion

from plotting import Histogram, FactorPlot

import time


class TargetColumnsTable(QtWidgets.QTableWidget):
    def __init__(self, data, window, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent=parent)
        
        self.window=window
        self.parent=parent
        
        self.verticalHeader().setDefaultSectionSize(48)
        
        # Table of features
        self.setRowCount(len(data))
        self.setColumnCount(5)
        
        self.setHorizontalHeaderLabels(data.columns)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        
        for i in range(len(data)):
            self.setItem(i, 0, TableNumericItem(str(i+1)))
            
            col = data['Column'].iloc[i]
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(str(col)))
            but = QtWidgets.QPushButton(col)
            but.setObjectName('table')
            but.clicked.connect(lambda x=col: self.window.ChangeTarget(x))
            self.setCellWidget(i, 1, but)
            
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(data['Type'].iloc[i]))
            
            uniques = data['Unique'].iloc[i]
            missing = data['Missing'].iloc[i]
            
            self.setItem(i, 3, TableNumericItem(str(uniques)))
            self.setCellWidget(i, 3, PercentBar(color='color_1', value=uniques, maxValue=data['Unique'].max()))
            
            self.setItem(i, 4, TableNumericItem(str(missing)))
            self.setCellWidget(i, 4, PercentBar(color='color_2', value=missing, probability=True))

        self.setSortingEnabled(True)
        
        # Stretch feature column
        for j in [3, 4]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
            
        for j in [0, 1, 2]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
        

class TargetSelectionFrame(QtWidgets.QWidget):
    # Contain options for new experiment
    def __init__(self, project, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        self.project = project
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        
        # Experiment options
        self.MakeOptionsBox()
        
        # Target Preview
        self.MakePreviewBox()
        
        # Table
        stats = self.project.get_column_summary()
        table = TargetColumnsTable(stats, self, parent=self)
        self.grid.addWidget(table, 1, 0, 1, 2)
        
        # Set default values
        self.ChangeTarget(self.project.get_all_columns()[0])
        self.ChangeValidationMethod('Cross Validation')
                
        # Layout columsn
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(1, 1)
    
    
    def MakeOptionsBox(self):
        box = BoxFrame(title='Choose your target')
        self.grid.addWidget(box, 0, 0)
        
        label = QtWidgets.QLabel('What would you like to predict?')
        box.grid.addWidget(label, 0, 0)
        
        self.targetBox = OptionsBox()
        self.targetBox.addItems(self.project.get_all_columns())
        self.targetBox.currentIndexChanged[str].connect(self.ChangeTarget)
        box.grid.addWidget(self.targetBox, 1, 0)
        
        label = QtWidgets.QLabel('Target data type')
        box.grid.addWidget(label, 2, 0)
        
        self.typeBox = OptionsBox()
        self.typeBox.currentIndexChanged[str].connect(self.ChangeType)
        box.grid.addWidget(self.typeBox, 3, 0)
        
        label = QtWidgets.QLabel('How would you like to score models?')
        box.grid.addWidget(label, 4, 0)
        
        self.metricBox = OptionsBox()
        self.metricBox.currentIndexChanged[str].connect(self.ChangeScorer)
        box.grid.addWidget(self.metricBox, 5, 0)

        label = QtWidgets.QLabel('Model validation method')
        box.grid.addWidget(label, 6, 0)
        
        opts = OptionsBox()
        opts.addItems(['Cross Validation', 'Train-Test Split'])
        opts.currentIndexChanged[str].connect(self.ChangeValidationMethod)
        box.grid.addWidget(opts, 7, 0)
        
        self.validationLabel = QtWidgets.QLabel('Number of folds')
        box.grid.addWidget(self.validationLabel, 8, 0)
        
        self.validationSlider = ValueSlider([3, 10], default=5)
        box.grid.addWidget(self.validationSlider, 9, 0)
        
        label = QtWidgets.QLabel('Scoring weights')
        box.grid.addWidget(label, 10, 0)
        
        self.weightsBox = OptionsBox()
        weights = [''] + self.project.get_non_negative_columns()
        self.weightsBox.addItems(weights)
        box.grid.addWidget(self.weightsBox, 11, 0)
        
        
    def MakePreviewBox(self):
        self.previewBox = BoxFrame(title='')
        self.grid.addWidget(self.previewBox, 0, 1)
        
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.canvas = MplFigureCanvas(self.fig)
        self.previewBox.grid.addWidget(self.canvas, 0, 0)
        self.canvas.hide()
        
        # Or table of dependence for numeric and categoric features
        self.resultsTable = QtWidgets.QTableWidget()
        self.previewBox.grid.addWidget(self.resultsTable, 0, 0)
        self.resultsTable.hide()
        
        # Or a warning for eg too many unique values
        self.targetWarning = QtWidgets.QLabel('')
        self.targetWarning.setObjectName('warning')
        self.previewBox.grid.addWidget(self.targetWarning, 0, 0)
        self.targetWarning.hide()
        
        
    def ShowSummary(self):
        self.resultsTable.hide()
        self.targetWarning.hide()
        self.canvas.hide()
            
        # Show preview
        if self.targetType == 'Numeric':
            self.canvas.show()
            
            self.fig.clear()
            Histogram(self.project.data, self.target, fig=self.fig)
            self.canvas.draw()
        else:
            counts = self.project.get_frequencies(self.target, 'categoric')
            
            if counts is not None:
                make_values_table(self.resultsTable, counts)
                self.resultsTable.show()
            else:
                self.targetWarning.setText('Too many unique values')
                self.targetWarning.show()
            
        self.previewBox.ChangeTitle(self.target)


    def ChangeTarget(self, target):
        self.target = target
        
        self.targetBox.setCurrentIndex(self.project.get_all_columns().index(target))
        
        # Update target types
        dtypes = ['Categoric']
        if target in self.project.get_numeric_columns():
            dtypes = ['Numeric', 'Categoric']
            
        self.typeBox.clear()
        self.typeBox.addItems(dtypes)
        
        self.targetType = dtypes[0]
        
        self.ShowSummary()


    def ChangeType(self, targetType):
        self.targetType = targetType
        
        # Update potential metrics
        mets = self.project.get_metrics(self.target, self.targetType, ui=True)
        
        self.metricBox.clear()
        self.metricBox.addItems(mets)
        
        self.ChangeScorer(mets[0])
        
        self.ShowSummary()


    def ChangeScorer(self, scorer):
        self.scorer = scorer
        
        
    def ToggleAdvancedOptions(self):
        if self.advancedShown:
            text = 'Show Advanced Options'
            self.advancedFrame.hide()
        else:
            text = 'Hide Advanced Options'
            self.advancedFrame.show()
        
        self.advancedShown = not self.advancedShown
        self.advancedBut.setText(text)
        
        
    def ChangeValidationMethod(self, method):
        self.validationMethod = method
        
        # Update options
        if method == 'Cross Validation':
            text = 'Number of folds'
            bounds = [3, 10]
            default = 5
        else:
            text = 'Test size %'
            bounds = [1, 50]
            default = 10
            
        self.validationLabel.setText(text)
        self.validationSlider.SetBounds(bounds)
        self.validationSlider.SetValue(default)
        
        
    def GetOptions(self):
        weights = self.weightsBox.currentText()
        if weights == '':
            weights = None
            
        #Turn scorers into sklearn usable names
        scorers={'Mean Squared Error': 'neg_mean_squared_error',
                 'Mean Absolute Error':'neg_mean_absolute_error',
                 'Median Absolute Error':'neg_median_absolute_error',
                 'Mean Squared Log Error': 'neg_mean_squared_log_error',
                 'R Squared (Recommended)':'r2',
                 'Accuracy':'accuracy',
                 'Log Loss (Recommended)':'neg_log_loss',
                 'F1 Score': 'f1',
                 'ROC Area Under Curve': 'roc_auc'}
        
        validation_size = self.validationSlider.value()
        if self.validationMethod == 'Train-Test Split':
            validation_size /= 100
        
        return {'target': self.target,
                'classification': self.targetType == 'Categoric',
                'scorer': scorers[self.scorer],
                'validation_method': self.validationMethod,
                'validation_size': validation_size,
                'weights': None}
        
        
class FeatureSelectionTable(QtWidgets.QTableWidget):
    def __init__(self, project, frame, target, classification, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent=parent)
        
        self.project=project
        self.frame=frame
        self.target=target
        self.parent=parent
        
        # Tracking values
        self.use = {}
        self.dtypes = {}
        
        cols = self.project.get_all_columns()
        nums = self.project.get_numeric_columns()
        stats = self.project.get_column_summary()
        
        self.verticalHeader().setDefaultSectionSize(48)
        
        # Table of features
        self.setRowCount(len(cols)-1)
        self.setColumnCount(6)
        
        self.setHorizontalHeaderLabels(['Use', '#', 'Feature', 'Type', 'Unique', 'Missing'])
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        
        row = 0
        for i in range(len(stats)):
            f = stats['Column'].iloc[i]
            
            if f == target:
                continue

            #Use
            use = QtWidgets.QCheckBox()
            use.setChecked(True)
            use.stateChanged.connect(lambda state, x=f: self.UseFeature(x))
            self.setCellWidget(row, 0, CenterWidget(use))
            
            self.use[f] = True

            # Number
            self.setItem(row, 1, TableNumericItem(str(i+1)))
            
            # Feature name     
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(f)))
            
            # Select
            but = QtWidgets.QPushButton(f)
            but.setObjectName('table')
            but.clicked.connect(lambda x=f: self.SelectFeature(x))
            self.setCellWidget(row, 2, but)
            
            # Feature type
            if f in nums:
                types = ['Numeric', 'Categoric']
            else:
                types = ['Categoric', 'Text']
                
            self.setItem(row, 3, QtWidgets.QTableWidgetItem(str(types[0])))
            self.dtypes[f] = types[0]
            
            frame = QtWidgets.QFrame()
            grid = QtWidgets.QGridLayout()
            grid.setContentsMargins(0,0,0,0)
            frame.setLayout(grid)
            
            typeBox = OptionsBox(parent=self)
            typeBox.setMaximumHeight(32)
            typeBox.addItems(types)
            typeBox.activated[str].connect(lambda value, y=f: self.ChangeType(y, value))
            
            grid.addWidget(typeBox, 0, 0)
            
            self.setCellWidget(row, 3, frame)
            
            # Unique and missing values
            uniques = stats['Unique'].iloc[i]
            missing = stats['Missing'].iloc[i]
            
            self.setItem(row, 4, TableNumericItem(str(uniques)))
            self.setCellWidget(row, 4, PercentBar(color='color_1', value=uniques, maxValue=stats['Unique'].max()))
            
            self.setItem(row, 5, TableNumericItem(str(missing)))
            self.setCellWidget(row, 5, PercentBar(color='color_2', value=missing, probability=True))

            row += 1


        self.setSortingEnabled(True)
        
        # Stretch feature column
        for j in [4, 5]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
            
        for j in [0, 1, 2, 3]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
            
            
    # Change type
    def ChangeType(self, feature, dtype):
        self.dtypes[feature] = dtype
        
        # Change value in table
        self.setSortingEnabled(False)
        self.setItem(self.GetRow(feature), 3, QtWidgets.QTableWidgetItem(str(dtype)))
        self.setSortingEnabled(True)
        
 
    # Select feature to summarise
    def SelectFeature(self, feature):
        self.frame.SelectFeature(feature, self.dtypes[feature])     
    
    
    # Toggle
    def UseFeature(self, feature):
        self.use[feature] = not self.use[feature]
        
    
    def GetFeatures(self):
        nums = []
        cats = []
        text = []
        
        for f in self.use:            
            if not self.use[f]:
                continue
            
            if self.dtypes[f] == 'Numeric':
                nums += [f]
            elif self.dtypes[f] == 'Categoric':
                cats += [f]
            elif self.dtypes[f] == 'Text':
                text += [f]
        
        return nums, cats, text
        
    
    def select_all(self):
        self.SelectFeaturesForUse('all')
    
    
    def deselect_all(self):
        self.SelectFeaturesForUse('none')
        

    def SelectFeaturesForUse(self, by):
        select = (by == 'all')
        for i in range(self.rowCount()):
            if self.item(i, 2).text() != str(self.target):
                self.cellWidget(i, 0).widget.setChecked(select)
                
                
    def GetRow(self, feature):
        # Find the row containing the feature
        for i in range(self.rowCount()):
            if self.item(i, 2).text() == str(feature):
                return i
  
        
class FeaturesSelectionFrame(QtWidgets.QWidget):
    # Contain options for new experiment
    def __init__(self, project, options, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        self.project = project
        self.target = options['target']
        self.options = options
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)

        box = BoxFrame('Select Features')
        self.grid.addWidget(box, 0, 0)
        
        self.table = FeatureSelectionTable(project, self, self.target, options['classification'])
        box.grid.addWidget(self.table, 0, 0, 1, 3)
        
        
        but = QtWidgets.QPushButton('Select All')
        but.clicked.connect(self.table.select_all)
        box.grid.addWidget(but, 1, 0)
        
        but = QtWidgets.QPushButton('Deselect All')
        but.clicked.connect(self.table.deselect_all)
        box.grid.addWidget(but, 1, 1)
        
        box.grid.setColumnStretch(2, 1)
        
        box.grid.setRowStretch(0, 1)
        box.grid.setColumnStretch(2, 1)
        
        self.grid.setRowStretch(0, 1)
        self.grid.setColumnStretch(0, 1)
        
        
    def MakePreviewBox(self):
        self.previewBox = BoxFrame(title='Select a feature below to preview')
        self.grid.addWidget(self.previewBox, 0, 0)
        
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.canvas = MplFigureCanvas(self.fig)
        self.previewBox.grid.addWidget(self.canvas, 0, 0)
        self.canvas.hide()

    
    def SelectFeature(self, feature, dtype):
        pass

#        # Show a preview of the targets dependence
#        xs_options = {feature: {'dtype': dtype.lower()}}
#        
#        target = self.options['target']
#        categoric = self.options['classification']
#        
#        if categoric:
#            ys = [None]
#            styles = ['frequency']
#        else:
#            ys = [None, target]
#            styles = ['frequency', 'mean']
#        
#        FactorPlot(self.project.data,
#                   xs=[feature],
#                   ys=ys,
#                   styles=styles,
#                   xs_options=xs_options,
#                   color=target,
#                   color_categoric=categoric,
#                   fig=self.fig)
#        
#        self.canvas.show()
#        self.canvas.draw()
        
        
    def GetOptions(self):
        nums, cats, text = self.table.GetFeatures()
        
        self.options['nums'] = nums
        self.options['cats'] = cats
        self.options['text'] = text
        
        return self.options


class NamingFrame(QtWidgets.QWidget):
    def __init__(self, project, options, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        
        self.project = project
        self.options = options
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        
        box = BoxFrame('Experiment Name')
        self.grid.addWidget(box, 1, 0)

        self.nameEdit = QtWidgets.QLineEdit('')
        box.grid.addWidget(self.nameEdit, 0, 0)
        
        self.nameWarning = QtWidgets.QLabel('')
        self.nameWarning.setObjectName('warning')
        box.grid.addWidget(self.nameWarning, 1, 0)
        
        box.grid.setRowStretch(2, 1)
        
        for i in [0, 2]:
            self.grid.setRowStretch(i, 1)
        
        
    def CheckName(self):
        """ Make sure the name is allowed """
        warn, safe = '', True
        name = self.nameEdit.text()
        
        if not self.project.check_experiment_name(name):
            warn, safe = 'Name taken', False
        if name == '':
            warn, safe = 'Enter a name for the experiment', False

        self.nameWarning.setText(warn)
        return safe
    
    
    def GetOptions(self):
        """ Return the name """
        self.options['name'] = self.nameEdit.text()
        
        return self.options
        
        
class NewExperimentFrame(QtWidgets.QWidget):
    def __init__(self, project, experimentsTab, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        self.setMaximumWidth(max_width())
        self.setContentsMargins(0,0,0,0)
        
        self.project = project
        self.experimentsTab = experimentsTab
        
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(0)
        self.setLayout(self.grid)
        
        # Current window
        self.step = 0
        self.windows = [TargetSelectionFrame(self.project, self)]
        self.grid.addWidget(self.windows[0], 0, 0)
        
        # Buttons
        bar = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(bar)
        grid.setContentsMargins(0,5,0,0)
        
        but = QtWidgets.QPushButton('Cancel')
        but.setObjectName('subtle')
        but.clicked.connect(self.Cancel)
        grid.addWidget(but, 0, 0)
        
        but = QtWidgets.QPushButton('Back')
        but.setObjectName('subtle')
        but.clicked.connect(self.Back)
        grid.addWidget(but, 0, 2)
        
        but = QtWidgets.QPushButton('Continue')
        but.clicked.connect(self.Continue)
        grid.addWidget(but, 0, 3)
        
        grid.setColumnStretch(1, 1)

        self.grid.addWidget(bar, 1, 0)
        self.grid.setRowStretch(0, 1)


    def Back(self):
        if self.step == 0:
            self.Cancel()
        else:
            # Remove window and show last
            self.step -= 1
            self.windows[-1].setParent(None)    
            self.windows = self.windows[:-1]
            self.windows[-1].show()
    
    
    def Continue(self):
        self.options = self.windows[-1].GetOptions()
        self.windows[-1].hide()
        
        if self.step == 0:
            # Define problem
            window = FeaturesSelectionFrame(self.project, self.options)
        elif self.step == 1:
            # Choose training options
            window = NamingFrame(self.project, self.options)
        elif self.step == 2:
            # Name the experiment
            if not self.windows[-1].CheckName():
                self.windows[-1].show()
                return

            self.experimentsTab.StartNewExperiment(self.options)
            return
        
        self.grid.addWidget(window, 0, 0)
        self.windows.append(window)
        
        self.step += 1
    
    
    def Cancel(self):
        # Stop and close this window
        self.experimentsTab.Back()
        
        
class ExperimentFrame(QtWidgets.QWidget):
    """ Frame containing list of models and summary frames
    """
    def __init__(self, project, experiment, experimentsTab):
        QtWidgets.QWidget.__init__(self)
        
        self.setMaximumWidth(max_width())
        
        self.project = project
        self.experiment = experiment
        self.experimentsTab = experimentsTab
        
        # Make the model summary tabs
        self.modelTabs = ModelTabs(self.experimentsTab, self.experiment)
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setColumnStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        
        # Training options
        box = BoxFrame(title='Training')
        self.grid.addWidget(box, 0, 0)
        
        for i in [0, 1]:
            box.grid.setColumnStretch(i, 1)
        
        box.grid.addWidget(QtWidgets.QLabel('Max training time (mins)'), 0, 0, 1, 2)
        
        self.timeBox = QtWidgets.QSpinBox()
        self.timeBox.setRange(1, 300)
        self.timeBox.setValue(10)
        box.grid.addWidget(self.timeBox, 1, 0, 1, 2)
        
        but = QtWidgets.QPushButton('Cancel All')
        but.setObjectName('subtle')
        but.clicked.connect(self.cancel_all)
        box.grid.addWidget(but, 2, 0)
        
        but = QtWidgets.QPushButton('Train All')
        but.clicked.connect(self.train_all)
        box.grid.addWidget(but, 2, 1)
        
        # Stats
        self.totalBox = LabelBox('Total Models Tested', '0')
        self.grid.addWidget(self.totalBox, 0, 1)
        
        self.trainingBox = LabelBox('Total Training Time', '0')
        self.grid.addWidget(self.trainingBox, 0, 2)
        
        self.UpdateStats()


        # Table of models
        box = BoxFrame(title='All Models')
        self.grid.addWidget(box, 1, 0, 1, 3)
        
        self.table = ModelsTable(experiment, self)
        box.grid.addWidget(self.table, 0, 0)
        

    def select_model(self, model):
        self.modelTabs.SelectModel(model)
        self.experimentsTab.OpenWindow(self.modelTabs, model)


    def TrainModel(self, model):
        max_train_time = 60 * float(self.timeBox.value())
        self.experimentsTab.TrainModel(self.experiment, model, max_train_time)
        
        
    def CancelFitting(self, model):
        self.experimentsTab.CancelFitting(self.experiment, model)
        
        
    def train_all(self):
        for model in self.experiment.models:
            self.TrainModel(model)
            
            
    def cancel_all(self):
        for model in self.experiment.models:
            self.CancelFitting(model)
            
            
    def UpdateStats(self):
        total, time = 0, 0
        
        for mod in self.experiment.models.values():
            total += len(mod.loss_results)
            time += max(mod.total_time + [0])
            
        self.totalBox.ChangeValue(str(total))
        self.trainingBox.ChangeValue(FormatTime(time))
        

    def update_model_status(self, experiment, model, status):
        if experiment != self.experiment:
            return
        
        self.table.update_model_status(model, status)
        self.modelTabs.update_model_status(model, status)
        
        # Also update the number of models tested and total training time
        self.UpdateStats()
        
        
class ModelsTable(QtWidgets.QTableWidget):
    def __init__(self, experiment, window, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent=parent)
        
        self.window = window
        self.experiment = experiment
        self.models = experiment.models
        
        self.verticalHeader().setDefaultSectionSize(48)
        
        # Table of features
        self.setRowCount(len(self.models))
        self.setColumnCount(6)
        
        if self.experiment.validation_method == 'random cv':
            title = 'CV Score'
        else:
            title = 'Test Score'
        
        self.setHorizontalHeaderLabels(['#', 'Model', title, 'Holdout Score', 'Status', 'Training'])
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        
        for i, model in enumerate(experiment.models):
            self.setItem(i, 0, TableNumericItem(str(i+1)))
            
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(str(model)))
            but = QtWidgets.QPushButton(model)
            but.setObjectName('table')
            but.clicked.connect(lambda x=model: self.window.select_model(x))
            self.setCellWidget(i, 1, but)
            
            if experiment.models[model].trained:
                self.setItem(i, 2, TableNumericItem(FormatScore(experiment.models[model].best_score_, scorer=experiment.scorer)))
                self.setItem(i, 3, TableNumericItem(FormatScore(experiment.models[model].holdout_score, scorer=experiment.scorer)))
            
            self.setItem(i, 5, QtWidgets.QTableWidgetItem('Train'))
            but = QtWidgets.QPushButton('Train')
            but.setObjectName('table')
            but.clicked.connect(lambda x=model: self.TrainModel(x))
            self.setCellWidget(i, 5, but)
            

        self.setSortingEnabled(True)
        
        # Stretch feature column
        for j in [4]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
            
        for j in [0, 1, 2, 3, 5]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
            
    def TrainModel(self, model):
        self.window.TrainModel(model)
        
        
    def CancelFitting(self, model):
        self.window.CancelFitting(model)

    
    def update_model_status(self, model, status):
        self.setSortingEnabled(False)
        
        i = self.get_model_row(model)
        self.setItem(i, 4, QtWidgets.QTableWidgetItem(status))
        
        # Update score
        if status == '':
            if self.experiment.models[model].trained:
                val_score = self.experiment.models[model].best_score_
                holdout_score = self.experiment.models[model].holdout_score
                
                self.setItem(i, 2, TableNumericItem(FormatScore(val_score, scorer=self.experiment.scorer)))
                self.setItem(i, 3, TableNumericItem(FormatScore(holdout_score, scorer=self.experiment.scorer)))
            
            self.setItem(i, 5, QtWidgets.QTableWidgetItem('Train'))
            but = self.cellWidget(i, 5)
            but.setText('Train')
            but.clicked.disconnect()
            but.clicked.connect(lambda x=model: self.TrainModel(x))
        else:
            self.setItem(i, 5, QtWidgets.QTableWidgetItem('Stop'))
            but = self.cellWidget(i, 5)
            but.setText('Stop')
            but.clicked.disconnect()
            but.clicked.connect(lambda x=model: self.CancelFitting(x))
            
        self.setSortingEnabled(True)

    
    def get_model_row(self, model):
        """ Returns the row of the table that belongs to a model"""
        for i in range(len(self.models)):
            if self.item(i, 1).text() == model:
                return i
        
        
class ModelTabs(QtWidgets.QTabWidget):
    """ Tabs of model details and summary
    """
    def __init__(self, experimentsTab, experiment, *args, **kwargs):
        """ Parameters
            experimentsTab: ExperimentsHomeTab
                the parent window
            experiment: Granite Experiment
                the experiment to show
            model: str
                the model to show     
        """
        
        QtWidgets.QTabWidget.__init__(self, *args, **kwargs)
        
        self.experimentsTab = experimentsTab
        self.experiment = experiment
        self.model = None
        
        # Make tabs
        self.addTab(TrainingFrame(self, experiment), 'Training')
        self.addTab(ModelDetailsFrame(self, experiment), 'Details')
        self.addTab(FeaturesFrame(self, experiment), 'Features')
        
        self.addTab(ImportanceFrame(self, experiment), 'Importance')
        self.addTab(InteractiveFrame(self, experiment), 'Interactive')
        
        if self.experiment.classification:
            self.addTab(ROCFrame(self, experiment), 'ROC Curve')
            self.addTab(ConfusionFrame(self, experiment), 'Confusion Matrix')
            
        self.addTab(PredictionFrame(self, experiment), 'Predictions')
        
            
    def SelectModel(self, model):
        """ Refresh tabs to show selected model. Hide non applicable tabs """
        
        self.model = model
        
        if not self.experiment.models[model].trained or self.experiment.models[model].queued:
            # Only show training frame
            self.setCurrentIndex(0)
            self.widget(0).SelectModel(model)
            
            for i in range(1, self.count()):
                self.setTabEnabled(i, False)
        else:
            for i in range(0, self.count()):
                
                # Only show importance tab when available
                if self.tabText(i) == 'Importance':
                    if self.experiment.get_feature_importance(model) is None:
                        self.setTabEnabled(i, False)
                        continue
                        
                self.setTabEnabled(i, True)
                self.widget(i).SelectModel(model)
                
        # Manually reset stylesheet to hide/show correct tabs
        self.setStyleSheet(app_css())
    
    
    def TrainModel(self, model, max_train_time):
        self.experimentsTab .TrainModel(self.experiment, model, max_train_time)
        
        
    def CancelTraining(self, model):
        self.experimentsTab.CancelFitting(self.experiment, model)
        
        
    def update_model_status(self, model, status):
        pass
    
    
class TrainingFrame(QtWidgets.QFrame):
    """ Shows training results and best score
    
    Parameters:
        modelTabs: ModelTabs frame
            the frame containing this
        experiment: Granite Experiment
            the experiment this frame is showing
        model: string
            current model name
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):        
        QtWidgets.QFrame.__init__(self, *args, **kwargs)

        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        self.model = None

        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        # Plot
        box = BoxFrame(title='Score Over Time')
        
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplFigureCanvas(self.fig)
        box.grid.addWidget(self.canvas, 0, 0)
        box.grid.setRowStretch(0, 1)
        self.grid.addWidget(box, 0, 1, 5, 1)
        
        # Training options
        self.trainBar = BoxFrame(title='Training')
        self.trainBar.setMinimumHeight(150)
        self.trainBar.grid.addWidget(QtWidgets.QLabel('Max training time (mins)'), 0, 0)
        self.timeBox = QtWidgets.QSpinBox()
        self.timeBox.setRange(1, 300)
        self.timeBox.setValue(10)
        self.trainBar.grid.addWidget(self.timeBox, 1, 0)
        
        but = QtWidgets.QPushButton('Train')
        but.clicked.connect(self.TrainModel)
        self.trainBar.grid.addWidget(but, 2, 0)
        
        self.grid.addWidget(self.trainBar, 0, 0)
        
        # Cancel training if running
        self.cancelBar = BoxFrame(title='Training')
        self.cancelBar.setMinimumHeight(150)
        self.cancelBar.grid.addWidget(QtWidgets.QLabel('Time remaining'), 0, 0)
        
        self.timeLeft = QtWidgets.QLabel('')
        self.cancelBar.grid.addWidget(self.timeLeft, 1, 0)
        
        but = QtWidgets.QPushButton('Stop')
        but.clicked.connect(self.CancelTraining)
        self.cancelBar.grid.addWidget(but, 2, 0)
        
        self.grid.addWidget(self.cancelBar, 0, 0)
        self.cancelBar.hide()

        # Information boxes
        self.bestScore = LabelBox('Best CV Score', '-')
        self.grid.addWidget(self.bestScore, 1, 0)
        
        self.parametersTested = LabelBox('Models Tested', '0')
        self.grid.addWidget(self.parametersTested, 2, 0)
        
        self.trainTime = LabelBox('Training Time', '0')
        self.grid.addWidget(self.trainTime, 3, 0)
        
        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(4, 1)
        
        self.thread = self.UpdateThread(self)
        self.thread.update_time.connect(self.UpdateTime)
        self.thread.update_plot.connect(self.UpdateResults)
        self.thread.start()
        
        self.wasFitting = False
        
        
    def TrainModel(self):
        """ Send signal to app to train this model """
        self.modelTabs.TrainModel(self.model_name, self.timeBox.value() * 60)
        
        # Hide other frames if it's training
        self.modelTabs.SelectModel(self.model_name)
    
    
    def CancelTraining(self):
        """ Send signal to app to cancel training for this model """
        self.modelTabs.CancelTraining(self.model_name)
        self.UpdateResults()
        self.UpdateTime()
    
    
    def SelectModel(self, model):
        self.model = self.experiment.models[model]
        self.model_name = model
        
        self.UpdateTime()
        self.UpdateResults()


    def UpdateTime(self):
        """ Update the training time, and cancel or start training buttons 
        based on model progress
        """
        


        if self.model.fitting or self.model.queued:
            self.cancelBar.show()
            self.trainBar.hide()

            if self.model.fitting:
                t = FormatTime(self.model.max_train_time + self.model.fit_start - time.time())
            else:
                t = 'In queue'

            self.timeLeft.setText(t)
            self.wasFitting = True
        else:
            # Fitting finished
            self.trainBar.show()
            self.cancelBar.hide()
            if self.wasFitting:
                self.wasFitting = False
                self.modelTabs.SelectModel(self.model_name)

        self.trainTime.ChangeValue(FormatTime(max(self.model.total_time + [0])))


    def UpdateResults(self):
        """ Update the best model score and training progress
        """
        pars_searched = len(self.model.loss_results)
        self.ax.clear()
        
        if pars_searched > 0:
            self.bestScore.ChangeValue(FormatScore(self.model.best_score_, self.experiment.scorer))
        else:
            self.bestScore.ChangeValue('-')

        self.parametersTested.ChangeValue(str(pars_searched))
        
        self.experiment.plot_training_scores(self.model_name, fig=self.fig, ax=self.ax)
        self.canvas.draw()
        
    
    class UpdateThread(QtCore.QThread):
        update_time = QtCore.Signal()
        update_plot = QtCore.Signal()
        
        def __init__(self, frame):
            QtCore.QThread.__init__(self)
            self.frame = frame
            self.total_runs = 0
            self.fitting = True
            
        def run(self):
            while True:
                time.sleep(0.1)
                if self.frame.model is not None:
                    if self.frame.model.fitting or self.frame.model.queued:
                        self.fitting = True
                        self.update_time.emit()
                        if len(self.frame.model.loss_results) != self.total_runs:
                            # Only plot when we get new info
                            self.total_runs = len(self.frame.model.loss_results)
                            self.update_plot.emit()
                    else:
                        # Update with new results
                        if self.fitting:
                            self.fitting = False
                            self.update_time.emit()
                            self.update_plot.emit()
                            
                            
class FeaturesFrame(QtWidgets.QFrame):
    """ Shows feature dependence on variables
    
    Parameters:
        modelTabs: ModelTabs frame
            the frame containing this
        experiment: Granite Experiment
            the experiment this frame is showing
        model: string
            current model name
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        
        self.grid = QtWidgets.QGridLayout(self)
        
        # Table of features
        box = BoxFrame(title='Features')
        self.grid.addWidget(box, 0, 0)
        self.table = QtWidgets.QTableWidget()
        box.grid.addWidget(self.table, 0, 0)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.setColumnCount(3)
        self.table.setRowCount(len(self.experiment.nums + self.experiment.cats + self.experiment.text))
        self.table.setHorizontalHeaderLabels(['#', 'Feature', 'Type'])
        self.table.verticalHeader().setDefaultSectionSize(48)
        
        # Stretch columns
        for j in [0, 2]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
        for j in [1]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)

        i = 0
        for f in self.experiment.data.columns:
            if f not in self.experiment.nums + self.experiment.cats + self.experiment.text:
                continue
            
            if f in self.experiment.nums:
                t = 'Numeric'
            elif f in self.experiment.cats:
                t = 'Categoric'
            else:
                t = 'Text'
            
            self.table.setItem(i, 0, TableNumericItem(str(i+1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(f)))
            
            but = QtWidgets.QPushButton(str(f))
            but.setObjectName('table')
            but.clicked.connect(lambda x=f: self.SelectFeature(x))
            self.table.setCellWidget(i, 1, but)
            
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(t))
            
            i += 1
            
        # Plots
        self.feature = None
        self.model = None
        self.bins = 25
        self.dataset = 'Train'
        if self.experiment.classification:
            cats = self.experiment.data[self.experiment.target].unique()
            self.cat = cats[0]
            self.ylims = [0, 1]
        else:
            cats = None
            self.cat = None
            self.ylims = [min(self.experiment.y), max(self.experiment.y)]

        
        
        # Plots of features
        self.options = PlotOptions(self, bins=True, cats=cats, dataSets=True, ylims=self.ylims)

        self.figureBox = BoxFrame(title='Select A Feature', side=self.options)
        self.figureBox.grid.setSpacing(1)
        self.grid.addWidget(self.figureBox, 0, 1)
        
        self.options.hide()
        
        self.fig = Figure(figsize=(4,4), dpi=72)
        self.canvas = MplFigureCanvas(self.fig)
        self.figureBox.grid.addWidget(self.canvas, 0, 1)
        self.canvas.hide()
        
        # Or table of dependence for numeric and categoric features
        self.resultsTable = QtWidgets.QTableWidget()
        self.figureBox.grid.addWidget(self.resultsTable, 0, 1)
        self.resultsTable.hide()
        
        self.grid.setColumnStretch(1, 1)
        self.figureBox.grid.setColumnStretch(1, 1)
        
        
    def ChangeBins(self, bins):
        self.bins = int(bins)
        self.SelectFeature(self.feature)
        
        
    def ChangeDataset(self, dataset):
        self.dataset = dataset
        self.SelectFeature(self.feature)
        

    def ChangeCat(self, cat):
        self.cat = cat
        self.SelectFeature(self.feature)
        
        
    def ChangeYLims(self, ymin=None, ymax=None):
        if ymin is not None:
            self.ylims[0] = ymin
        if ymax is not None:
            self.ylims[1] = ymax
            
        self.SelectFeature(self.feature)
        

    def SelectModel(self, model):
        self.model = model
        
        
    def SelectFeature(self, feature):
        if feature is None or self.model is None:
            return
        
        self.figureBox.ChangeTitle(str(feature))
        
        self.options.show()
        
        self.feature = feature
        if feature in self.experiment.nums:
            # Plot if numeric
            self.experiment.plot_feature_dependence(feature, model=self.model,
                                                    cat=self.cat, train=self.dataset=='Train',
                                                    bins=self.bins, ylims=self.ylims,
                                                    fig=self.fig)
            self.canvas.draw()
            
            self.canvas.show()
            self.resultsTable.hide()
            
            self.options.ToggleBins(hide=False)
        else:
            # Else make a table
            stats, counts, bins = self.experiment.get_feature_dependence(feature, model=self.model,
                                                                         cat=self.cat, train=self.dataset=='Train')
            
            if feature in self.experiment.cats:
                name = 'Category'
            else:
                name = 'Word'

            if self.cat is None:
                ave = 'Mean'
            else:
                ave = 'Proportion'

            cols = [name, 'Count', ave, 'Predicted']
                
            df = pd.DataFrame(columns=cols)

            df[name] = bins
            df['Count'] = counts
            df[ave] = stats['y'].values
            df['Predicted'] = stats['ypred'].values
            
            # Fill table
            make_values_table(self.resultsTable, df,
                              probability=self.experiment.classification,
                              ymin=self.ylims[0], ymax=self.ylims[1])  
            
            self.canvas.hide()
            self.resultsTable.show()
            self.options.ToggleBins(hide=True)
            
            
class ImportanceFrame(QtWidgets.QFrame):
    """ Shows feature importance
    
    Parameters:
        modelTabs: ModelTabs frame
            the frame containing this
        experiment: Granite Experiment
            the experiment this frame is showing
        model: string
            current model name
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        
        self.grid = QtWidgets.QGridLayout(self)
        
        # Table of features
        box = BoxFrame(title='Feature Importance')
        self.grid.addWidget(box, 0, 0)
        self.table = QtWidgets.QTableWidget()
        box.grid.addWidget(self.table, 0, 0)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.setColumnCount(2)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(['Feature', 'Importance'])
        self.table.verticalHeader().setDefaultSectionSize(48)
        
        # Stretch columns
        for j in [0]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
        for j in [1]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
        

    def SelectModel(self, model):
        self.model = model
        
        fi = self.experiment.get_feature_importance(model)

        if fi is None:
            return
        
        self.table.setRowCount(len(fi))
        
        # Add to table
        self.table.setSortingEnabled(False)
        for i in range(len(fi)):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem('{:}'.format(fi['Feature'].iloc[i])))
            self.table.setItem(i, 1, TableNumericItem('{:}'.format(fi['Importance'].iloc[i])))
            self.table.setCellWidget(i, 1, PercentBar(value = fi['Importance'].iloc[i], probability=True))
            
        self.table.setSortingEnabled(True)
            
  

class ReasonsTable(QtWidgets.QTableWidget):
    """ Table showing the most important factors determining a prediction """
    def __init__(self, features, center, maxAbs, parent=None):
        
        QtWidgets.QTableWidget.__init__(self, parent=parent)

        self.setSortingEnabled(False)
        self.setRowCount(0)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['#', 'Feature', 'Difference'])
        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(48)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        for j in [0]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
        for j in [1, 2]:
            self.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
            
        self.setSortingEnabled(True)

        
    def UpdateValues(self, difference, center, maxAbs, probability=False):
        self.setSortingEnabled(False)
        
        n = min(5, len(difference))
        self.setRowCount(n)
        
        difference['Abs'] = difference['Difference'].abs()
        top = difference.nlargest(n, 'Abs')
        
        for i in range(n):
            self.setItem(i, 0, QtWidgets.QTableWidgetItem('{:}'.format(i+1)))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem('{:}'.format(top['Feature'].iloc[i])))
            self.setItem(i, 2, TableNumericItem(FormatNumber(top['Difference'].iloc[i])))
            self.setCellWidget(i, 2, ValueBar(value=top['Difference'].iloc[i], center=center, maxAbs=maxAbs, probability=probability))
                    
        self.setSortingEnabled(True)


class InteractiveFrame(QtWidgets.QFrame):
    """ Allows users to get live predictions
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        
        self.grid = QtWidgets.QGridLayout(self)

        # Table of features
        self.MakeFeatureTable()

        # Results table for probability estimates by class
        if self.experiment.classification:
            box = BoxFrame(title='Predicted Probabilities')
            self.grid.addWidget(box, 0, 1)
            
            self.resultsTable = QtWidgets.QTableWidget()
            self.resultsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.resultsTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            self.resultsTable.verticalHeader().hide()
            self.resultsTable.setSortingEnabled(True)
            self.resultsTable.setColumnCount(2)
            self.resultsTable.setHorizontalHeaderLabels(['Category', 'Probability'])
            
            for j in [0, 1]:
                self.resultsTable.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
                
            box.grid.addWidget(self.resultsTable, 0, 0)
        else:
            box = BoxFrame(title='Predicted Value')
            box.setMinimumHeight(128)
            self.grid.addWidget(box, 0, 1)
            
            self.percentBar = PercentBar(color='color_3',
                                         value=0,
                                         minValue=self.experiment.y.min(),
                                         maxValue=self.experiment.y.max(),
                                         label=True)
            
            box.grid.addWidget(self.percentBar, 0, 0)
            box.grid.setRowStretch(0, 1)
            box.grid.setColumnStretch(0, 1)


        self.currentModel = [None, None]

        # Reasons table
        box = BoxFrame(title='Top Reasons')
        self.grid.addWidget(box, 1, 1)
        self.reasonsTable = ReasonsTable(self.nums+self.cats+self.text, 0, 1)
        box.grid.addWidget(self.reasonsTable, 0, 0)

        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        
        
    def MakeFeatureTable(self):
        """ Make the feature with input variables """
        
        # Different input types
        self.nums, self.cats, self.text = self.experiment.nums, self.experiment.cats, self.experiment.text
        
        self.lastType = {}
        self.lineEdits = {}

        # Input data
        self.data = pd.DataFrame()
        
        box = BoxFrame('Input Features')
        self.grid.addWidget(box, 0, 0, 2, 1)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['#', 'Feature', 'Type', 'Value'])
        box.grid.addWidget(self.table, 0, 0)
        self.table.verticalHeader().setDefaultSectionSize(48)
        
        # Stretch columns
        for j in [0, 1, 2]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
            
        for j in [3]:
            self.table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
            
        self.table.setRowCount(len(self.nums+self.cats+self.text))
        
        i = 0
        for f in self.experiment.data.columns:
            if f not in self.nums + self.cats + self.text:
                continue
            
            self.table.setItem(i, 0, TableNumericItem(str(i+1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(f)))
            
            if f in self.nums:
                t = 'Numeric'
            elif f in self.cats:
                t = 'Categoric'
            else:
                t = 'Text'
                
            self.lastType[f] = t
                
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(t))
            
            # Allow users to change values
            if t == 'Numeric':
                
                w = NumericEdit(bounds=[self.experiment.data[f].min(), self.experiment.data[f].max()],
                                command=lambda value, y=f: self.ChangeValue(y, value),
                                default=self.experiment.data[f].mean())
                
                # Take average value by default
                self.data[f] = [self.experiment.data[f].mean()]
                
            elif t == 'Categoric':

                # With a dropdown menu
                cb = OptionsBox()
                
                values = list(self.experiment.data[f].unique())
                
                if np.nan in values:
                    values.remove(np.nan)
                    
                values = sorted(values)
                
                # Add in an unknown value
                cb.addItem('', userData=np.nan)
            
                for v in values:
                    cb.addItem(str(v), userData=v)
                    
                values = [np.nan] + list(values)
                    
                self.data[f] = [np.nan]
                cb.setCurrentIndex(0)
                cb.currentIndexChanged.connect(lambda i, x=values, y=f: self.ChangeValue(y, x[i]))

                cb.setMaximumHeight(32)
                
                w = QtWidgets.QFrame()
                grid = QtWidgets.QGridLayout(w)
                grid.setContentsMargins(0,0,0,0)
                
                grid.addWidget(cb, 0, 0)
            else:
                # With a line edit
                w = QtWidgets.QLineEdit()
                w.returnPressed.connect(lambda y=f: self.ChangeTextValue(y))
                w.editingFinished.connect(lambda y=f: self.ChangeTextValue(y))
                
                self.data[f] = ['']
                self.lineEdits[f] = w
            
            self.table.setCellWidget(i, 3, w)
            
            i += 1
            
            
    def SelectModel(self, model_name):
        self.model = self.experiment.models[model_name]
        self.model_name = model_name
            
    
    def ChangeValue(self, f, value):
        self.data[f] = [value]
        self.GetPrediction()
        
        
    def ChangeTextValue(self, f):
        text = self.lineEdits[f].text()
        self.ChangeValue(f, text)
        
        
    def GetPrediction(self):
        #Get predictions - nothing here for dummy 
        if len(self.data) == 0:
            self.data['null'] = [0]

        pred, diffs, probs = self.experiment.get_prediction_reasons(self.model_name, self.data)

        if self.experiment.classification:
            pred = pd.DataFrame()
            pred['Category'] = self.model.best_estimator_.classes_
            pred['Probability'] = list(probs)
            
            make_values_table(self.resultsTable,
                              pred,
                              probability=self.experiment.classification)
            
            maxAbs = 1

        else:
            self.percentBar.ChangeValue(pred,
                                        minValue=self.experiment.y.min(),
                                        maxValue=self.experiment.y.max())
            
            maxAbs = self.experiment.y.max() - self.experiment.y.min()

        self.reasonsTable.UpdateValues(diffs, 0, maxAbs, probability=self.experiment.classification)
        
        
class PipelineDiagram(QtWidgets.QFrame):
    """ Plots the model pipeline and lets users click on each component """
    def __init__(self, pipeline, parent=None):
        """ Parameters:
            pipeline: sklearn.pipeline.Pipeline
                model pipeline to be shown
            controller: Qt
        """
        
        QtWidgets.QFrame.__init__(self)
        
        self.grid = QtWidgets.QGridLayout(self)
        self.pipeline = pipeline

        self.ShowPipe()


    def ShowPipe(self):
        # Connections for arrows
        self.cons = {}
        self.buts = {}

        # Initial button
        height = self.PipeHeight(self.pipeline)
        self.buts[0] = QtWidgets.QPushButton('Data')
        self.grid.addWidget(self.buts[0], height-1, 0)
        
        # Connection map
        for i in range(self.PipeSize(self.pipeline) + 1):
            self.cons[i] = []
        
        self.MakePipe(self.pipeline, 1, height-1, 1, 0, '')

        for j in range(2 * self.PipeLength(self.pipeline) + 1):
            self.grid.setColumnStretch(j, 1)


    def MakePipe(self, step, x0, y0, count, source, step_id):
        """ A recursive function to create the pipeline diagram
        Arguments:
            step: tuple 
                step in the pipeline
            x0: int
                start of the pipe
            y0: int
                center height of the pipe or subpipe
            count: int
                count of node id
            source: int
                leading node id
            step_id: str
                name of step in pipeline eg 'pre__nums'
        """
        if type(step) == FeatureUnion:
            y = y0 - (self.PipeHeight(step) - 1)
            # Make each branch
            self.cons[source] = []
            for i, split in enumerate(step.transformer_list):
                self.MakePipe(split[1], x0, y, count, source, step_id+'__'+split[0])
                y += 2 * self.PipeHeight(split)
                
                count += self.PipeSize(split[1])
                
                # Merge
                self.cons[count-1] = [source + self.PipeSize(step)]

            # Make the merge node
            self.buts[count] = QtWidgets.QPushButton('Merge')
            self.grid.addWidget(self.buts[count], y0, 2 * (x0 + self.PipeLength(step)-1))
            
        elif type(step) == Pipeline:
            for sub in step.steps:
                self.MakePipe(sub[1], x0, y0, count, source, step_id+'__'+sub[0])
                x0 += self.PipeLength(sub[1])
                count += self.PipeSize(sub[1])
                source = count-1
                
        else:
            name = self.GetName(step)
            self.buts[count] = QtWidgets.QPushButton(name)
            self.grid.addWidget(self.buts[count], y0, 2 * x0)
            self.cons[source] += [count]
        

    def paintEvent(self, event):
        for c in self.cons:
            for d in self.cons[c]:
                self.DrawArrow(self.buts[c], self.buts[d])
        
        
    # Draw a bezier curve from the right of 1 to the left of 2
    def DrawArrow(self, w1, w2):
        # Draw line between buttons
        path = QtGui.QPainterPath()
        
        x1, y1 = w1.x() + w1.width(), w1.y() + w1.height()/2
        x2, y2 = w2.x(), w2.y() + w2.height()/2
        
        path.moveTo(x1, y1)
        path.cubicTo((x1 + x2)/2, y1, (x1 + x2)/2, y2, x2, y2)
        
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        
        pen = QtGui.QPen(QtGui.QColor(59,104,185), 2)
        qp.setPen(pen)

        qp.drawPath(path)
        
        # Draw arrow head
        size = 5
        points = [QtCore.QPointF(x2, y2), QtCore.QPointF(x2-size, y2-size),
                  QtCore.QPointF(x2-size, y2+size), QtCore.QPointF(x2, y2)]
        
        tri = QtGui.QPolygonF(points)
        
        brush = QtGui.QBrush(QtGui.QColor(59,104,185))
        qp.setBrush(brush)
        qp.drawPolygon(tri)
        
    

    def PipeHeight(self, transformer):
        if type(transformer) == FeatureUnion:
            return sum([self.PipeHeight(t[1]) for t in transformer.transformer_list])
        elif type(transformer) == Pipeline:
            return max([self.PipeHeight(t[1]) for t in transformer.steps])
        else:
            return 1
    
    
    def PipeLength(self, transformer):
        if type(transformer) == FeatureUnion:
            # +1 for the merge node
            return max([self.PipeLength(t[1]) for t in transformer.transformer_list])+1
        elif type(transformer) == Pipeline:
            return sum([self.PipeLength(t[1]) for t in transformer.steps])
        else:
            return 1
        
        
    def PipeSize(self, transformer):
        if type(transformer) == FeatureUnion:
            # +1 for the merge node
            return sum([self.PipeSize(t[1]) for t in transformer.transformer_list])+1
        elif type(transformer) == Pipeline:
            return sum([self.PipeSize(t[1]) for t in transformer.steps])
        else:
            return 1
        
        
    def DescribeStep(self, step):
        name = str(step.__class__)
        
        if 'xgboost' in name:
            return 'Source: XGBoost'
        
        elif 'sklearn' in name:
            return 'Source: scikit-learn'
        
        else:
            return 'Source: Granite AI'
        
        
    def GetName(self, step):
        if type(step) == str:
            return step
        
        return step.__class__.__name__
        
        

class ModelDetailsFrame(QtWidgets.QFrame):
    """ Shows the model pipeline and best parameters
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')

        self.modelTabs = modelTabs
        self.experiment = experiment
        self.model = None
        
        self.grid = QtWidgets.QGridLayout(self)
        
        
    def SelectModel(self, model):
        self.model = model
        
        mod = self.experiment.models[model]
        
        # Show model pipeline
        self.pipeFrame = BoxFrame('Model Pipeline')
        self.pipeFrame.setMaximumWidth(max_width())
        pipe = PipelineDiagram(mod.best_estimator_)
        self.pipeFrame.grid.addWidget(pipe)
        self.grid.addWidget(self.pipeFrame, 1, 1)
        
        # Show best parameters
        self.bestPars = BoxFrame('Best Parameters')
        self.grid.addWidget(self.bestPars, 1, 2)

        row = self.ShowBestParameters(mod.best_estimator_, prefix='', row=0, pars=mod.best_params_)
        self.bestPars.grid.setRowStretch(row, 1)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 98)
        self.grid.setColumnStretch(3, 1)
        
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(2, 1)
        
        
    def ShowBestParameters(self, step, prefix, row, pars):
        """ A recursive function to create the best parameters for each step
        Arguments:
            step: tuple
                step in the pipeline
            prefix: str
                the prefix for the step eg 'pre__' for preprocessing
            row: int
                the current row to print things to
            pars: dict
                dictionary of best parameters
        """

        if type(step) == FeatureUnion:
            # Make each branch
            for split in step.transformer_list:
                row += self.ShowBestParameters(split[1], prefix + split[0] + '__', row, pars)
            
        elif type(step) == Pipeline:
            # Make each step
            for sub in step.steps:
                row += self.ShowBestParameters(sub[1], prefix + sub[0] + '__', row, pars)
        else:
            # Make the step
            sub_pars = [par for par in pars if prefix in par]
            
            if len(sub_pars) == 0:
                return row
            
            # Title
            title = QtWidgets.QLabel(step.__class__.__name__)
            title.setObjectName('subtitle')
            self.bestPars.grid.addWidget(title, row, 0)
            self.bestPars.grid.addWidget(StyleLine(), row+1, 0, 1, 2)
            row += 2
            
            # Parameters
            for par in sub_pars:
                self.bestPars.grid.addWidget(QtWidgets.QLabel(par.split('__')[-1]), row, 0)
                self.bestPars.grid.addWidget(QtWidgets.QLabel(str(pars[par])), row, 1)
                row += 1
                
            # Empty row
            self.bestPars.grid.addWidget(QtWidgets.QLabel(''), row, 0)
            self.bestPars.grid.addWidget(QtWidgets.QLabel(''), row, 1)
            
            row += 1
                
        return row
            

class PlotOptions(QtWidgets.QFrame):
    """ Contains dropdown options for model summary plots eg which dataset 
    
    Parameters:
    
        cats: list, optional
            list of categories to show at a time eg only show probability in
            on of the cateogories
        bins: bool, optional
            number of bins in histogram
        dataSets: bool, optional
            choose between train and holdout datasets
        xrange: [min, max], optional
            control x range of plot
        yrange: [min, max], optional
            control y range of plot
    """
    
    def __init__(self, parent, cats=None, bins=False, dataSets=False, xlims=None, ylims=None):
        QtWidgets.QFrame.__init__(self)
        
        self.parent_frame = parent
        
        grid = QtWidgets.QVBoxLayout(self)
        
        self.cats = cats
        
        # Track the numeric bins options so we can hide them when showing a cat
        self.bin_widgets = []
        
        # Number of bins when getting histograms
        if bins:
            label = QtWidgets.QLabel('Bins')
            label.setObjectName('subtitle')
            self.bin_widgets = [label]
            grid.addWidget(self.bin_widgets[-1])
            
            binBox = OptionsBox()
            binBox.addItems(['5', '10', '25', '50', '100'])
            binBox.setCurrentIndex(2)
            binBox.currentIndexChanged[str].connect(self.parent_frame.ChangeBins)
            self.bin_widgets.append(binBox)
            grid.addWidget(self.bin_widgets[-1])

        #Category options
        if cats is not None:
            label = QtWidgets.QLabel('Category')
            label.setObjectName('subtitle')
            grid.addWidget(label)
            
            self.catBox = OptionsBox()   
            for cat in sorted(list(cats)):
                self.catBox.addItem(str(cat), userData=cat)

            self.catBox.currentIndexChanged.connect(self.ChangeCat)
            grid.addWidget(self.catBox)
            
        # Train or test results?
        if dataSets:
            label = QtWidgets.QLabel('Data Set')
            label.setObjectName('subtitle')
            grid.addWidget(label)
            
            self.dataBox = OptionsBox(parent=self)
            self.dataBox.addItems(['Train', 'Holdout'])
            self.dataBox.currentIndexChanged[str].connect(self.parent_frame.ChangeDataset)
            grid.addWidget(self.dataBox)
            
        # Min and maximum values
        if xlims is not None:
            label = QtWidgets.QLabel('X Limits')
            label.setObjectName('subtitle')
            grid.addWidget(label)
            
            grid.addWidget(QtWidgets.QLabel('Minimum'))
            self.minXBox = NumericLineEdit(xlims[0], command=lambda x: self.parent_frame.ChangeXLims(xmin=x))
            grid.addWidget(self.minXBox)
            
            grid.addWidget(QtWidgets.QLabel('Maximum'))
            self.maxXBox = NumericLineEdit(xlims[1], command=lambda x: self.parent_frame.ChangeXLims(xmax=x))
            grid.addWidget(self.maxXBox)
        
        if ylims is not None:
            label = QtWidgets.QLabel('Y Limits')
            label.setObjectName('subtitle')
            grid.addWidget(label)
            
            grid.addWidget(QtWidgets.QLabel('Minimum'))
            self.minYBox = NumericLineEdit(ylims[0], command=lambda y: self.parent_frame.ChangeYLims(ymin=y))
            grid.addWidget(self.minYBox)
            
            grid.addWidget(QtWidgets.QLabel('Maximum'))
            self.maxYBox = NumericLineEdit(ylims[1], command=lambda y: self.parent_frame.ChangeYLims(ymax=y))
            grid.addWidget(self.maxYBox)
            
        # Stretch last row to fill
        grid.addStretch()
        
        
    def ToggleBins(self, hide=True):
        """ Hide or show the bins options. Used to track options but we want to
        hide bins because of a categorical feature
        """
        
        for w in self.bin_widgets:
            if hide:
                w.hide()
            else:
                w.show()
                
                
    def ChangeCat(self, index):
        cat = self.catBox.itemData(index)
        self.parent_frame.ChangeCat(cat)
    
    
class ROCFrame(QtWidgets.QFrame):
    """ Shows ROC Curve and AUC for classifiers
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        self.model = None
        self.dataset = 'Train'
        
        self.grid = QtWidgets.QGridLayout(self)
        
        opts = PlotOptions(self, dataSets=True)
        
        box = BoxFrame(side = opts)
        box.setMaximumWidth(max_width())
        box.grid.setSpacing(1)
        self.grid.addWidget(box, 0, 0)

        self.fig = Figure(figsize=(4,4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplFigureCanvas(self.fig)
        box.grid.addWidget(self.canvas, 0, 0)


    def SelectModel(self, model):
        self.model = model
        self.Replot()
        
        
    def ChangeDataset(self, dataset):
        self.dataset = dataset
        self.Replot()
    
    
    def Replot(self):
        self.ax.clear()
        self.experiment.plot_roc_curve(self.model, self.dataset=='Train',
                                       self.fig, self.ax)
        
        self.canvas.draw()
        
        
class ConfusionFrame(QtWidgets.QFrame):
    """ Shows confusion matrix for classifiers
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        self.model = None
        
        self.grid = QtWidgets.QGridLayout(self)
        self.setLayout(self.grid)
        
        opts = PlotOptions(self, dataSets=True)
        box = BoxFrame(side=opts)
        box.setMaximumWidth(max_width())
        box.grid.setSpacing(1)
        self.grid.addWidget(box, 0, 0)

        self.fig = Figure(figsize=(4,4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplFigureCanvas(self.fig)
        box.grid.addWidget(self.canvas, 0, 0)

        # Set used to plot
        self.dataset = 'Train'
        
        
    def SelectModel(self, model):
        self.model = model
        self.Replot()
        
        
    def ChangeDataset(self, dataset):
        self.dataset = dataset
        self.Replot()
    
    
    def Replot(self):
        self.ax.clear()
        self.experiment.plot_confusion_matrix(self.model, self.dataset=='Train',
                                              self.fig, self.ax)
        
        self.canvas.draw()
            
    
class PredictionFrame(QtWidgets.QFrame):
    """ Allows users to get bactch predictions
    """
    def __init__(self, modelTabs, experiment, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        self.setObjectName('background')
        
        self.modelTabs = modelTabs
        self.experiment = experiment
        self.model = None
        
        self.grid = QtWidgets.QGridLayout(self)
        
        box = BoxFrame(title='Make Predicions')
        self.grid.addWidget(box, 0, 0)
        
        box.grid.addWidget(QtWidgets.QLabel('Predict On'), 0, 0)
        
        opts = OptionsBox()
        opts.addItems(['Training Data', 'New Data'])
        opts.activated[str].connect(self.GetPredictions)
        box.grid.addWidget(opts, 1, 0)
        
        self.saveBut = QtWidgets.QPushButton('Save')
        self.saveBut.clicked.connect(lambda: self.SavePredictions())
        self.saveBut.setEnabled(False)
        box.grid.addWidget(self.saveBut, 2, 0)
        
        # Blank widget for now
        self.tableBox = BoxFrame('Predictions')

        self.grid.addWidget(self.tableBox, 0, 1, 2, 1)
                
        self.table = QtWidgets.QWidget()
        self.tableBox.grid.addWidget(self.table, 0, 0)
        
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(1, 1)
    
    
    def SelectModel(self, model):
        self.model = model
        
        # Hide any previous predictions
        self.saveBut.setEnabled(False)
        self.table.setParent(None)
        self.table = QtWidgets.QWidget()
        self.grid.addWidget(self.table, 2, 0, 1, 5)
        
    
    def GetPredictions(self, dataset):
        if dataset == 'Training Data':
            data = None
        else:
            data = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]     
            if data == '':
                return
        
        self.preds = self.experiment.get_predictions(model=self.model, data=data, append=True)
        
        # Remake table
        self.table.setParent(None)
        self.table = PagedTable(self.preds)
        self.tableBox.grid.addWidget(self.table, 2, 0, 1, 5)
        
        self.saveBut.setEnabled(True)
        
        
    def SavePredictions(self):
        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File')[0]     
        if name == '':
            return
        
        self.preds.to_csv(name, index=False)
            
        
class ExperimentIcon(BoxFrame):
    """ The icon to select an experiment with """
    def __init__(self, experiment, name):
        BoxFrame.__init__(self, name)

        label = QtWidgets.QLabel('Target: ' + str(experiment.target))
        self.grid.addWidget(label, 0, 0)
    
        self.grid.setColumnStretch(0, 1)
        self.button = QtWidgets.QPushButton('Open')
        self.grid.addWidget(self.button, 0, 1)

        
            
class ExperimentsTab(WindowManager):
    """ The tab containing all experiments """
    def __init__(self, project, parent=None):
        self.project = project
        self.itemsFrame = ItemsView(self)
        self.itemsFrame.setMaximumWidth(600)
        
        for exp in self.project.experiments:
            # Add to the items frame
            self.AddItem(exp)

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setObjectName('background')
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.itemsFrame)
        self.scrollArea.setMaximumWidth(600)

        WindowManager.__init__(self, self.scrollArea, 'Experiments')

        self.experimentFrame = None

        # Get the training thread going
        self.train_thread = self.TrainThread(self)
        self.train_thread.status_changed.connect(self.update_queue_status)
        self.train_thread.trained.connect(self.update_training_status)
        self.train_thread.start()


    def NewItem(self):
        # Make a new experiment
        self.OpenWindow(NewExperimentFrame(self.project, self), 'New Experiment')
        
        
    def AddItem(self, experiment):
        widget = ExperimentIcon(self.project.experiments[-1], self.project.experiments[-1].name)
        self.itemsFrame.AddItem(widget, widget.button)
        
        
    def StartNewExperiment(self, options):
        # Close the model making window and add experiment
        self.project.make_new_experiment(options)
        self.Back()
        self.AddItem(self.project.experiments[-1])


    def SaveExperiment(self, style, options):
        pass


    def SelectItem(self, index):
        # Open that experiment and update training queue
        self.experimentFrame = ExperimentFrame(self.project, self.project.experiments[index], self)
        self.OpenWindow(self.experimentFrame, self.project.experiments[index].name)
        self.update_queue_status()
        

    def TrainModel(self, experiment, model, max_train_time=600):
        # Don't add if already in queue
        for [exp, mod, train_time] in self.train_thread.queue:
            if exp == experiment and mod == model:
                return
        
        self.train_thread.queue.append([experiment, model, max_train_time])
        experiment.models[model].queued = True
        self.update_queue_status()
        
        
    def CancelFitting(self, experiment, model):
        self.train_thread.CancelFitting(experiment, model)
        
        
    def update_queue_status(self):
        if self.experimentFrame is None:
            return
        
        # Update status of training queue
        for i, [exp, mod, train_time] in enumerate(self.train_thread.queue):
            if i == 0:
                self.experimentFrame.update_model_status(exp, mod, 'Training')
            else:
                self.experimentFrame.update_model_status(exp, mod, 'Position in queue: {:}'.format(i))
                
                
    def update_training_status(self, experiment, model):
        self.experimentFrame.update_model_status(experiment, model, '')
        

    class TrainThread(QtCore.QThread):
        status_changed = QtCore.Signal()
        trained = QtCore.Signal(object, object, object)
        
        def __init__(self, window):
            QtCore.QThread.__init__(self)
            self.queue = []
            
            
        def run(self):
            while True:
                if len(self.queue) == 0:
                    time.sleep(1)
                    continue

                [experiment, model, train_time] = self.queue[0]
                
                # Fit
                experiment.fit_model(model, train_time)
                
                # Update models tabs
                experiment.models[model].queued = False
                self.trained.emit(experiment, model, '')

                # Remove from queue
                if len(self.queue) > 0:
                    if [experiment, model] == self.queue[0][:2]:
                        self.queue.pop(0)
                    
                self.status_changed.emit()


        def CancelFitting(self, experiment, model):
            i = None
            for i, [exp, mod, train_time] in enumerate(self.queue):
                if exp == experiment and mod == model:
                    exp.models[mod].cancel_training()
                    self.queue.pop(i)
                    experiment.models[model].queued = False
            
            # Models in training will update once the holdout score is ready
            self.trained.emit(experiment, model, '')

            self.status_changed.emit()