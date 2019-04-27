# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 21:27:56 2018

@author: Mike Staddon
"""

import os
import pandas as pd

os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets
from qtpy.QtWidgets import QSizePolicy

from .widgets import (PagedTable, StyleLine, WindowManager, BoxFrame,
                     CenterWidget)

from .style import max_width, max_height


class LoadDataFrame(QtWidgets.QWidget):
    """ Starting a new project - naming and loading data
    """
    def __init__(self, app, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        # Supported filetypes
        self.fileTypes = ['.csv']
        
        self.app = app
        self.setMaximumWidth(max_width())
        
        
        self.grid = QtWidgets.QGridLayout(self)

        box = BoxFrame(title='Open A Data Set')
        self.grid.addWidget(box, 0, 0)
        
        box.grid.addWidget(QtWidgets.QLabel('File Path'), 0, 0, 1, 2)
        
        self.dataEdit = QtWidgets.QLineEdit()

        self.dataEdit.setText('')
        box.grid.addWidget(self.dataEdit, 2, 0)

        but = QtWidgets.QPushButton('...')
        but.setObjectName('subtle')
        but.clicked.connect(self.SelectFile)
        box.grid.addWidget(but, 2, 1)
        
        l = 'Supported Formats:' + ''.join(' ' + ftype for ftype in self.fileTypes)
        
        label = QtWidgets.QLabel(l)
        box.grid.addWidget(label, 3, 0, 1, 2)

        self.dataWarning = QtWidgets.QLabel()
        self.dataWarning.setObjectName('warning')
        box.grid.addWidget(self.dataWarning, 4, 0, 1, 2)
        
        but = QtWidgets.QPushButton('Preview Data')
        but.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        but.clicked.connect(self.PreviewData)
        but = CenterWidget(but)
        box.grid.addWidget(but, 5, 0)
        
        # Table showing data
        self.previewBox = BoxFrame('Data Preview')
        self.grid.addWidget(self.previewBox, 1, 0, 1, 1)
        self.previewBox.hide()
        
        # Confirm or cancel buttons
        frame = QtWidgets.QWidget()
        self.grid.addWidget(frame, 2, 0)
        
        grid = QtWidgets.QGridLayout(frame)
        grid.setContentsMargins(0, 0, 0, 0)
        
        but = QtWidgets.QPushButton('Cancel')
        but.setObjectName('subtle')
        but.clicked.connect(self.Cancel)
        grid.addWidget(but, 0, 0)
        
        but = QtWidgets.QPushButton('Confirm')
        but.clicked.connect(self.Confirm)
        grid.addWidget(but, 0, 2)
        
        grid.setColumnStretch(1, 1)
        
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 1)
        

    def SelectFile(self):
        # Select dataset using the file browser
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]     
        if (name == ''):
            return
        
        self.dataEdit.setText(name)
        self.CheckFile()

    
    def OpenFile(self, nrows=None):
        """ Open the file, either all rows or the first few for a preview """
        if not self.CheckFile():
            return
        
        name = self.dataEdit.text()
        
        try:
            if name[-4:] == '.csv':
                return pd.read_csv(name, nrows=nrows)
        except:
            self.dataWarning.setText('File type not supported. Only .csv ma')
            
            return None


    def CheckFile(self):
        """ Check that it has the right format """
        safe = True
        warning = ''
        
        name = self.dataEdit.text()
        
        #See if file exists
        if not os.path.exists(name):
            safe = False
            warning = 'File not found'
            
        # See if it has the correct format
        elif len(name) < 4 or name[-4:] not in self.fileTypes:
            safe = False
            warning = 'File type not supported'

        
        self.dataWarning.setText(warning)
        return safe
    
    
    def PreviewData(self):
        data = self.OpenFile(nrows=25)
        if data is None:
            return
        
        self.dataTable = PagedTable(data)
        self.previewBox.grid.addWidget(self.dataTable, 0, 0)
        self.previewBox.show()
    
    
    def Confirm(self):
        data = self.OpenFile()
        if data is None:
            return
            
        self.app.StartProject('', data)


    def Cancel(self):
        self.app.CancelNewProject()

        

class RecentProjectFrame(QtWidgets.QFrame):
    """ Starting a new project - naming and loading data
    """
    def __init__(self, projects, frame, *args, **kwargs):
        """
        Parameters
            projects: list
                list of recent projects
            frame: ProjectsFrame
                the parent projectsframe
        """
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        
        if projects is None:
            projects = []
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        label = QtWidgets.QLabel('Recent Projects')
        self.grid.addWidget(label, 0, 0)
        
        # Make the recents scrollable
        scrollArea = QtWidgets.QScrollArea()
        
        recentFrame = QtWidgets.QFrame()
        grid = QtWidgets.QGridLayout()
        recentFrame.setLayout(grid)

        # List of recent projects
        for i, p in enumerate(projects):
            text = p['name'] + '\n'+p['path']
            but = QtWidgets.QPushButton(text)
            but.clicked.connect(lambda x=i: frame.OpenProject(x))
            grid.addWidget(but, i, 0)

        scrollArea.setWidget(recentFrame)
        
        self.grid.addWidget(scrollArea, 1, 0)
        
        self.grid.setColumnStretch(0, 1)
        self.grid.setRowStretch(1, 1)


class ProjectsFrame(QtWidgets.QWidget):
    """ Frame containing recent projects and new project button
    """
    def __init__(self, app, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        self.app = app
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setSpacing(32)

        but = QtWidgets.QPushButton('New\nProject')
        but.clicked.connect(self.NewProject)
        self.grid.addWidget(but, 1, 1)
        
        but = QtWidgets.QPushButton('Open\nProject')
        but.clicked.connect(self.OpenProject)
        self.grid.addWidget(but, 1, 2)


        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(3, 1)
        
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(2, 1)


    def NewProject(self):
        self.app.NewProject()


    def OpenProject(self, index=None):
        self.app.OpenProject()
