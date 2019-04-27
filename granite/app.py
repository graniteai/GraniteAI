# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:43:00 2018

@author: Mike Staddon

Contains the parent UI window, and project view components
"""

import os
import sys

os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets


from .gui.projects import ProjectsFrame, LoadDataFrame
from .gui.visualisation import VisualisationTab
from .gui.experiments import ExperimentsTab
from .gui.style import app_css
from .gui.widgets import (PagedTable, HorizontalTabWidget, BoxFrame,
                         WindowManager)

from .ml import GraniteProject

import pickle
        
        
class DataTab(QtWidgets.QFrame):
    def __init__(self, project, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        
        self.setObjectName('background')
        self.project = project
        
        self.grid = QtWidgets.QGridLayout(self)
        
        box = BoxFrame(title='Data')
        
        table = PagedTable(self.project.data)
        box.grid.addWidget(table, 0, 0)
        
        self.grid.addWidget(box, 0, 0)
    
        

class GraniteApp(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        # Size
        self.setMinimumSize(900,766)
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(2)
        
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 1)
        
        self.project = None
        
        # Set the style
        self.setStyleSheet(app_css())
        
        # Top bar
        bar = QtWidgets.QFrame()
        bar.setObjectName('saveBar')
        grid = QtWidgets.QGridLayout(bar)
        grid.setSpacing(20)
        
        but = QtWidgets.QPushButton('New')
        but.setObjectName('saveBar')
        but.clicked.connect(self.NewProject)
        grid.addWidget(but, 0, 0)
        
        but = QtWidgets.QPushButton('Open')
        but.setObjectName('saveBar')
        but.clicked.connect(self.OpenProject)
        grid.addWidget(but, 0, 1)
        
        but = QtWidgets.QPushButton('Save')
        but.setObjectName('saveBar')
        but.clicked.connect(self.SaveProject)
        grid.addWidget(but, 0, 2)
        
        grid.setColumnStretch(3, 1)
        
        self.grid.addWidget(bar, 0, 0)
        
        # Projects window
        self.window = WindowManager(ProjectsFrame(self), 'Home')
        self.grid.addWidget(self.window, 1, 0)


    
    def NewProject(self):
        if not self.CheckSave():
            return

        self.window.setParent(None)
        self.window = WindowManager(LoadDataFrame(self), 'New Project')
        self.grid.addWidget(self.window, 1, 0)
        
        
    def CancelNewProject(self):
        self.window.setParent(None)
        self.window = WindowManager(ProjectsFrame(self))
        self.grid.addWidget(self.window, 1, 0)
    
    
    def StartProject(self, name, data):
        # Confirm a new project and show it
        self.project = GraniteProject(data, name)
        self.ShowProject(self.project)
        
        
    def ShowProject(self, project):
        self.window.setParent(None)
        
        self.project = project
        
        self.window = HorizontalTabWidget()
#        self.window = QtWidgets.QTabWidget()
        self.grid.addWidget(self.window, 1, 0)
        
        # Make data, visualisation, and experiments frame
        self.dataTab = DataTab(self.project)
        self.window.addTab(self.dataTab, 'Data')
        
        self.visTab = VisualisationTab(self.project)
        self.window.addTab(self.visTab, 'Visualisation')
        
        self.expTab = ExperimentsTab(self.project, parent=self)
        self.window.addTab(self.expTab, 'Experiments')
    
    
    def OpenProject(self):
        # Check if want to save previous project
        if not self.CheckSave():
            return
        
        openName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]  
        if (openName == ''):
            return
        
        load_data = pickle.load(open(openName, 'rb'))
        
        if type(load_data) != GraniteProject:
            # TO DO: Show warning
            print('Wrong type')
            return
        
        self.ShowProject(load_data)
    
    
    def SaveProject(self):
        if self.project is None:
            return
        
        save_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Open File')[0]  
        if (save_name == None or save_name == ''):
            return
        
        pickle.dump(self.project, open(save_name, 'wb'))
        

    def CheckSave(self):
        if self.project is None:
            return True
        
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText('')
        msgBox.setInformativeText('Would you like to save your current project?')
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
        
        ret = msgBox.exec_()
        
        # Reply
        if ret == QtWidgets.QMessageBox.Save:
            self.SaveProject()
            return True
        elif ret == QtWidgets.QMessageBox.No:
            return True
        elif ret == QtWidgets.QMessageBox.Cancel:
            return False
        else:
            # should never be reached
            return False


#if __name__ == '__main__':    
#    app = QtWidgets.QApplication(sys.argv)
#    app.setStyle('fusion')
#    ex = GraniteApp()
#    ex.show()
#    app.exec_()