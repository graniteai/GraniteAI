# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:00:00 2018

@author: Mike Staddon

Custom widgets to be used by other files
"""

import os
os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets, QtCore, QtGui

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

### Useful functions

def FormatScore(score, scorer):
    
    # Most maxmimise the negative of the loss
    if scorer != 'r2':
        score = abs(score)
        
    return FormatNumber(score)


# Format seconds into hrs:mins:secs as a string
def FormatTime(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    
    return '{:.0f}h:{:.0f}m:{:.0f}s'.format(h, m, s)


# Generic number formatting
def FormatNumber(x, probability=False, rng=1):
    """ Format a float to a string
    
    Arguments:
        x: float
            number to format
        probability: bool
            if True, show a percentage
        rng: float?
            range of value, shows more sf if the range is small
    """
    
    # 5 significant figures
    if probability:
        return '{:.2f} %'.format(100*x)
    
    # Return range + 5sf
    return '{:.5g}'.format(round(x, int(np.log10(rng))+5))


def make_values_table(table, data, probability=False, ymin=0, ymax=1):
    """ Files a QTableWidget with value bars from a pandas table
    
    Arguments:
        table: QTableWidget
            table to update
        data: dataframe
            data to put in, should be just be category and word, vs frequency 
            and value
        probability: bool, optional
            if True show percentages instead of values
        ymin: float, optional
            minimum value of value bars ie if the value is less than this the 
            bar will be empty
        ymax: float, optional
            maximum value of value bars
            
    """

    table.setSortingEnabled(False)
    table.setRowCount(len(data))
    table.setColumnCount(len(data.columns))
    table.setHorizontalHeaderLabels(data.columns)
    table.setShowGrid(False)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(48)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    
    # Fill table
    for i in range(len(data)):
        j = 0
        for c in data.columns:
            if c == 'Category' or c == 'Word':
                table.setItem(i, j, QtWidgets.QTableWidgetItem('{:}'.format(data[c][i])))
                
            elif c == 'Count':
                table.setItem(i, j, TableNumericItem('{:.0f}'.format(data[c][i])))
                table.setCellWidget(i, j, PercentBar('color_1', data[c][i], 0, data[c].max()))
                
            elif c == 'Mean' or c == 'Proportion' or c == 'Predicted' or c == 'Probability' or c == 'Partial':
                table.setItem(i, j, TableNumericItem(FormatNumber(data[c][i])))
                col = 'color_{:}'.format(j)                
                table.setCellWidget(i, j, PercentBar(col, data[c][i], ymin, ymax, probability=probability))
                
            j += 1

    
    for j in range(len(data.columns)):
        table.horizontalHeader().setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch)
        
    table.setSortingEnabled(True)

### Useful widgets

class StyleLine(QtWidgets.QFrame):
    """ A thin under line to highlight things """
    def __init__(self, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        
        self.setMinimumHeight(1)
        self.setMaximumHeight(1)
        

# Custon combobox to prevent scrolling through options...
class OptionsBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        QtWidgets.QComboBox.__init__(self, *args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
    # Disable scrolling
    def wheelEvent(self, event):
        return
    
# Wrapper class for the Figure Canvas - makes resizing charts nicer
class MplFigureCanvas(FigureCanvas):
    def __init__(self, *args, **kwargs):
        FigureCanvas.__init__(self, *args, **kwargs)

        
    def resizeEvent(self, event):
        FigureCanvas.resizeEvent(self, event)
        self.figure.tight_layout()
        self.draw()
        

class AddRemoveButton(QtWidgets.QPushButton):
    def __init__(self, text, add=True, *args, **kwargs):
        QtWidgets.QPushButton.__init__(self, text, *args, **kwargs)
        
#        self.setMaximumWidth(24)
#        
#        if add:
#            css = """
#            color: #3B68B9;
#            background: #FFFFFF;
#            font: bold;
#            """
#        else:
#            css = """
#            color: #FF882B;
#            background: #FFFFFF;
#            font: bold;
#            """
            
#        self.setStyleSheet(css)
        

class ItemsView(QtWidgets.QWidget):
    """ Frame containing save and new items eg projects, plots or experiments
    """
    def __init__(self, frame, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        """
        Load previous project plots
        
        Frame: QtWidget
            controllor of the items
        """
        
        self.frame = frame

        self.widgets = []
        
        self.grid = QtWidgets.QGridLayout(self)
        
        but = QtWidgets.QPushButton('New')
        but.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        but.clicked.connect(self.NewItem)
        self.addButton = CenterWidget(but)
        self.grid.addWidget(self.addButton, 0, 0)
        
        
    def sizeHint(self):
        return QtCore.QSize(100 * (1 + len(self.widgets)), 100)
        
        
    def NewItem(self):
        # Hit the new button
        self.frame.NewItem()
    
    
    def AddItem(self, widget, button, index=None):
        """Add an item to the list, push the new button down
        
        Parameters:
            widget: WtWidget
                the widget to display
            button: QPushButton
                button that opens this items contents
            index: int, optional
                place in list to insert widget, if None put at end
        """
        if index is None:
            # Add a new item
            self.widgets.append(widget)
            index = len(self.widgets)-1
            
            self.grid.addWidget(self.addButton, len(self.widgets), 0)
            
            # Stretch end
            self.grid.setRowStretch(len(self.widgets), 0)
            self.grid.setRowStretch(len(self.widgets)+1, 1)
        else:
            # Save over an old item
            self.widgets[index].setParent(None)
            self.widgets[index] = widget
            
        self.grid.addWidget(widget, index, 0)
        button.clicked.connect(lambda x=index: self.SelectItem(x))
        
        
        

    def SelectItem(self, index):
        self.frame.SelectItem(index)
        
        
class ValueSlider(QtWidgets.QWidget):
    """ A slider that shows the value next to it"""
    def __init__(self, bounds=None, default=0, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        
        if bounds is None:
            bounds = [0, 1]

        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.ValueChanged)
        grid.addWidget(self.slider, 0, 0)
        
        self.label = QtWidgets.QLabel()
        grid.addWidget(self.label, 0, 1)
        
        self.SetBounds(bounds)
        self.SetValue(default)


    def SetBounds(self, bounds):
        self.bounds = bounds
        self.slider.setMinimum(bounds[0])
        self.slider.setMaximum(bounds[1])
        

    def SetValue(self, value):
        self.slider.setValue(value)
        
        
    def ValueChanged(self):
        self.label.setText(str(self.slider.value()))
        
        
    def value(self):
        return self.slider.value()


class TableNumericItem(QtWidgets.QTableWidgetItem):
    def __init__(self, *args, **kwargs):
        QtWidgets.QTableWidgetItem.__init__(self, *args, **kwargs)
        # Right align numbers
        self.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
    
    def __lt__(self, other):
        if ( isinstance(other, QtWidgets.QTableWidgetItem) ):
            my_value = self.data(QtCore.Qt.EditRole)
            other_value = other.data(QtCore.Qt.EditRole)

            try:
                return float(my_value) < float(other_value)
            except:
                pass
            
        return super(TableNumericItem, self).__lt__(other)
    
        
class ValueBar(QtWidgets.QFrame):
    """ A combo of a label and a bar. Displays a value, and a bar filled scale
    """
    def __init__(self, colorNeg=None, colorPos=None, value=0, center=0, maxAbs=1, parent=None, label=True, probability=False, *args, **kwargs):
        """ Parameters:
            colorNeg: str
                hex color code, for values below the center
            colorPos: str
                hex color code, for values above the center
            value: float
                initial value of bar
            center: float
                center point of the bar, highlights values different to this
            maxAbs: float
                maximum difference from center that will be shown
            parent: QtWidget
                parent widget
            label: bool
                whether or not to write the value above the bar
            probability: bool
                if true, show percentage
        """
        QtWidgets.QFrame.__init__(self, parent=parent, *args, **kwargs)
        
        self.probability = probability
        
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        self.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(0)
        
        self.grid.setRowStretch(0, 2)
        self.grid.setRowStretch(1, 1)
        self.grid.setRowStretch(2, 1)
        
        if colorNeg is None:
            colorNeg = 'color_2'
            
        if colorPos is None:
            colorPos = 'color_1'
        
        self.hasLabel = label
        
        if self.hasLabel:
            self.label=QtWidgets.QLabel(FormatNumber(value, self.probability, rng=maxAbs))
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.grid.addWidget(self.label, 0, 0, 1, 4)
        
        f = QtWidgets.QFrame()
        f.setObjectName('background')
        self.grid.addWidget(f,1,0)
        
        f = QtWidgets.QFrame()
        f.setObjectName(colorNeg)
        self.grid.addWidget(f,1,1)
        
        f = QtWidgets.QFrame()
        f.setObjectName(colorPos)
        self.grid.addWidget(f,1,2)
        
        f = QtWidgets.QFrame()
        f.setObjectName('background')
        self.grid.addWidget(f,1,3)
        
        self.ChangeValue(value, center, maxAbs)
        
    
    def ChangeValue(self, value, center=0, maxAbs=1):
        """ Update the value, center or max """
        if self.hasLabel:
            text = FormatNumber(value, self.probability, rng=maxAbs)
            
            if value >= 0:
                text = '+'+text
                
            self.label.setText(text)
            
        self.grid.setColumnStretch(0, 100-int(max((center-value)/maxAbs * 100, 0)))
        self.grid.setColumnStretch(1, int(max((center-value)/maxAbs * 100, 0)))
        
        self.grid.setColumnStretch(2, int(max((value-center)/maxAbs * 100, 0)))
        self.grid.setColumnStretch(3, 100-int(max((value-center)/maxAbs * 100, 0)))
        

class PercentBar(QtWidgets.QFrame):
    """ A combo of a label and a bar. Displays a value, and a bar filled scale
    """
    def __init__(self, color=None, value=0, minValue=0, maxValue=1, parent=None, label=True, probability=False, *args, **kwargs):
        """
        Parameters:
            color: str
                hex color code, for values value bar
            value: float
                initial value of bar
            minValue: float
                value at which bar is empty
            maxValue: float
                value at which bar is full
            parent: QtWidget
                parent widget
            label: bool
                whether or not to write the value above the bar
            probability: bool
                if true, show percentage
        """
        QtWidgets.QFrame.__init__(self, parent=parent, *args, **kwargs)
        
        self.probability = probability
        
        self.grid = QtWidgets.QGridLayout(self)
        self.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(0)
        
        self.grid.setRowStretch(0, 2)
        self.grid.setRowStretch(1, 1)
        self.grid.setRowStretch(2, 1)
        
        self.hasLabel = label
        
        if self.hasLabel:
            self.label=QtWidgets.QLabel(FormatNumber(value))
            self.grid.addWidget(self.label, 0, 0, 1, 2)
        
        if color is None:
            color = 'color_1'
            
        f = QtWidgets.QFrame()
        f.setObjectName(color)
        self.grid.addWidget(f,1,0)
        
        f = QtWidgets.QFrame()
        f.setObjectName('background')
        self.grid.addWidget(f,1,1)
        
        self.ChangeValue(value, minValue, maxValue)
        
    
    def ChangeValue(self, value, minValue=0, maxValue=1):
        """ Update the value, center or max """
        if self.hasLabel:
            self.label.setText(FormatNumber(value, probability=self.probability))
            
        self.grid.setColumnStretch(0, int((value-minValue)/(maxValue - minValue) * 100))
        self.grid.setColumnStretch(1, 100-int((value-minValue)/(maxValue - minValue) * 100))
        
        
class NumericEdit(QtWidgets.QWidget):
    """ Combines a slider with a line edit, which also shows the value """
    def __init__(self, bounds, command=None, default=None):
        """Parameters
            bounds: tuple
                maximum and minimum values on the slider
            command: function, optional
                function to call on value update
            default: float, optional
                starting value. If none given then take the mid bound
        """
        QtWidgets.QWidget.__init__(self)
        
        self.setStyleSheet('padding: 0px; margin: 0px;')
        
        self.bounds = bounds
        self.command = command
        
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(0)
        grid.setContentsMargins(0,0,0,0)
        
        grid.setRowStretch(0, 1)
        for i in range(2):
            grid.setColumnStretch(i, 1)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setStyleSheet('padding: 0px; margin: 0px;')
        # Use percentage - convert to actual values on change
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        grid.addWidget(self.slider, 0, 0)
        
        self.lineEdit = QtWidgets.QLineEdit()
        self.lineEdit.setStyleSheet('padding: 0px; margin: 0px;')
        grid.addWidget(self.lineEdit, 0, 1)
        
        if default is None:
            default = sum(bounds)/2
            
        self.slider.setValue((default-self.bounds[0])/(self.bounds[1]-self.bounds[0])*100)
        self.lineEdit.setText(FormatNumber(default))
        self.value = default
        
        # Keep values in sync
        self.slider.sliderMoved.connect(self.SliderUpdateValue)
        self.slider.sliderReleased.connect(self.SliderFinishValue)
        self.lineEdit.editingFinished.connect(self.LineEditUpdateValue)
        self.lineEdit.returnPressed.connect(self.LineEditUpdateValue)


    def setValue(self, value):
        self.lineEdit.setText(FormatNumber(value))
        self.slider.setValue((value-self.bounds[0])/(self.bounds[1]-self.bounds[0])*100)
        

    def SliderUpdateValue(self):
        value = self.slider.value()*(self.bounds[1]-self.bounds[0])/100 + self.bounds[0]
        self.lineEdit.setText(FormatNumber(value))        
        self.value=value


    def SliderFinishValue(self):
        self.SliderUpdateValue()
        
        if self.command != None:
            self.command(self.value)
            
            
    def LineEditUpdateValue(self):
        try:
            value = float(self.lineEdit.text())
        except:
            self.lineEdit.setText('0')
            value = 0
            
        self.slider.setValue((value-self.bounds[0])/(self.bounds[1]-self.bounds[0])*100)   
        self.value = value
        
        if self.command != None:
            self.command(value)
            
            
class NumericLineEdit(QtWidgets.QLineEdit):
    """ A line edit that only takes numeric values
    
    Parameters:
        value: float
            default value
        parent: QtWidget, optional
            widget parent
        command: function, optional
            perform this function with the value on edit
    """
    
    def __init__(self, value, parent=None, command=None):
        QtWidgets.QLineEdit.__init__(self, parent=parent)
        
        self.value = value
        self.command = command
        
        self.setText(FormatNumber(value))
        
        self.editingFinished.connect(self.CheckValue)
        self.returnPressed.connect(self.CheckValue)

        
    def CheckValue(self):
        value = self.text()
        try:
            self.value = float(value)
            if self.command is not None:
                self.command(self.value)
        except:
            self.setText(str(value))
            
            
class PagedTableModel(QtCore.QAbstractTableModel):
    """ Table model to show pandas Dataframe data. Usese pages to limit the
    number of rows displayed at once """
    ROWS_PER_PAGE = 100
    
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        
        # Use numpy values! Muuuuuuch quicker than pandas for indexing
        self.data_ = data.values
        self.columns_ = data.columns
        self.current_page = 0
        self.max_page = (len(data) - 1) // self.ROWS_PER_PAGE


    def rowCount(self, parent=None):
        return min(len(self.data_)-1,
                   (self.current_page + 1) * self.ROWS_PER_PAGE -1) % self.ROWS_PER_PAGE + 1


    def columnCount(self, parent=None):
        return len(self.columns_)


    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                row = index.row() % self.ROWS_PER_PAGE + self.current_page * self.ROWS_PER_PAGE
                return str(self.data_[row, index.column()])
        return None


    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.columns_[col]
        return None
    
    
    def sort(self, Ncol, order):
        # Sort table by given column number
        self.layoutAboutToBeChanged.emit()
        
        column = self.data_.columns[Ncol]
        ascending = not order == QtCore.Qt.DescendingOrder
        
        self.data_.sort_values(column, ascending=ascending, inplace=True)
            
        self.layoutChanged.emit()
        
    
    def ChangePage(self, forward=False, back=False):
        new_page = self.current_page
        
        if forward:
            new_page += 1
        
        if back:
            new_page -= 1
            
        new_page = max(min(new_page, self.max_page), 0)
        
        if new_page == self.current_page:
            return
        
        self.current_page = new_page
        self.layoutChanged.emit()
        
        
class PagedTableView(QtWidgets.QTableView):
    """ The table view for paged pandas table model """
    def __init__(self, data):
        """
        Parameters
            data: pandas Dataframe
                data to show
        """
        QtWidgets.QTableView.__init__(self)
        
        self.model = PagedTableModel(data)
        self.setModel(self.model)
        self.verticalHeader().hide()
        self.setShowGrid(False)
        
            
            
class PagedTable(QtWidgets.QWidget):
    """ A table showing limitted rows, and buttons for pressing next. Used for
    showing large pandas frames without massive memory overhead """
    def __init__(self, data):
        """
        Parameters
            data: pandas Dataframe
                data to show
        """

        QtWidgets.QWidget.__init__(self)

        self.grid = QtWidgets.QGridLayout(self)
        
        # Table of data
        self.table = PagedTableView(data)
        self.grid.addWidget(self.table, 0, 0, 1, 5)
        
        # Next and previous buttons
        self.prev_but = QtWidgets.QPushButton('<')
        self.prev_but.clicked.connect(self.Previous)
        self.grid.addWidget(self.prev_but, 1, 1)
        
        self.next_but = QtWidgets.QPushButton('>')
        self.next_but.clicked.connect(self.Next)
        self.grid.addWidget(self.next_but, 1, 3)
        
        self.pageLabel = QtWidgets.QLabel('')
        self.grid.addWidget(self.pageLabel, 1, 2)
        
        self.grid.setRowStretch(0, 1)
        for i in [0, 4]:
            self.grid.setColumnStretch(i, 1)
            
        self.UpdatePageInfo()


    def Previous(self):
        self.table.model.ChangePage(back=True)
        self.UpdatePageInfo()
        

    def Next(self):
        self.table.model.ChangePage(forward=True)
        self.UpdatePageInfo()
        
        
    def UpdatePageInfo(self):
        # Check whether we can still go back or forward
        if self.table.model.current_page == self.table.model.max_page:
            self.next_but.setEnabled(False)
        else:
            self.next_but.setEnabled(True)
            
        if self.table.model.current_page == 0:
            self.prev_but.setEnabled(False)
        else:
            self.prev_but.setEnabled(True)
            
        info = 'Page {:d} / {:d}'.format(self.table.model.current_page+1, self.table.model.max_page+1)
        self.pageLabel.setText(info)        
        
class WindowManager(QtWidgets.QFrame):
    """ Manages parent and child windows. Opening a new window makes a next
    layer which users can press back to return to the previous window
    
    Arguments:
        frame: QtWidget
            the base frame window
    """
    
    def __init__(self, frame, name=None, *args, **kwargs):
        
        QtWidgets.QFrame.__init__(self, *args, **kwargs)

        self.setObjectName('background')
        self.frames = [frame]
        
        if name is None:
            name = ''
            
        self.names = [name]
        
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(0)
        
        self.navFrame = QtWidgets.QFrame()
        self.navFrame.setObjectName('navBar')
        self.grid.addWidget(self.navFrame, 0, 0)
        
        grid = QtWidgets.QGridLayout(self.navFrame)
        
        self.backBut = QtWidgets.QPushButton('Back')
        self.backBut.setObjectName('navBar')
        self.backBut.clicked.connect(self.Back)
        grid.addWidget(self.backBut, 0, 0)
        self.backBut.hide()
        
        self.windowLabel = QtWidgets.QLabel(name)
        self.windowLabel.setObjectName('navBar')
        grid.addWidget(self.windowLabel, 0, 2)
        
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        
        # Empty window - putting frames in here allows them to center
        self.window = QtWidgets.QWidget()
        self.windowGrid = QtWidgets.QGridLayout(self.window)
        
        self.grid.addWidget(self.window, 1, 0)
        self.grid.setRowStretch(1, 1)
        
        self.windowGrid.addWidget(frame, 0, 0)
        self.windowGrid.setContentsMargins(0,0,0,0)


    def OpenWindow(self, widget, name):
        self.frames[-1].hide()
        self.frames.append(widget)
        self.names.append(name)
        self.windowGrid.addWidget(widget, 1, 0)
        self.windowLabel.setText(name)
        self.backBut.show()


    def Back(self):
        if len(self.frames) == 1:
            return
        
        self.names.pop()
        widget = self.frames.pop()

        widget.setParent(None)
        self.frames[-1].show()
        self.windowLabel.setText(self.names[-1])
        
        if len(self.frames) == 1:
            self.backBut.hide()
            
            
# Horizontal tabs
#   http://www.riverbankcomputing.com/pipermail/pyqt/2005-December/011724.html
class HorizontalTabBar(QtWidgets.QTabBar):
    def __init__(self, parent=None, *args, **kwargs):
        self.tabSize = QtCore.QSize(kwargs.pop('width', 128), kwargs.pop('height', 64))
        QtWidgets.QTabBar.__init__(self, parent, *args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_StyleSheet)
         
        
    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionTab()
        
        # Style
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)

        pen = QtGui.QPen(QtGui.QColor('#FFFFFF'))
        painter.setPen(pen)
        
        for index in range(self.count()):
            self.initStyleOption(option, index)
            tabRect = self.tabRect(index)
            tabRect.moveLeft(10)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabShape, option)
            painter.drawText(tabRect, QtCore.Qt.AlignVCenter | QtCore.Qt.TextDontClip, self.tabText(index))

        
    def tabSizeHint(self, index):
        return self.tabSize
        size = QtWidgets.QTabBar.tabSizeHint(self, index)
        if size.width() < size.height():
            size.transpose()
        return size


class HorizontalTabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args, **kwargs):
        QtWidgets.QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(HorizontalTabBar())
        self.setTabPosition(QtWidgets.QTabWidget.West)
        self.setAttribute(QtCore.Qt.WA_StyleSheet)
        
        
class BoxFrame(QtWidgets.QFrame):
    """ A frame with a title and main body as in eg linkedin 
    
    Use .grid to access grid for lower layout
    
    Parameters:
        title: str
            title above box, if None, don't include the first row
        side: QWidget
            widget to go to the side of main frame eg plot options
    
    """
    def __init__(self, title=None, side=None, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)
        
        # Adds border lines around the main widgets
        self.setObjectName('boxFrame')
        
        grid = QtWidgets.QGridLayout(self)
        
        # Adds a border to windows
        grid.setContentsMargins(1,1,1,1)
        grid.setSpacing(1)
        
        
        # Only make the title if needed
        if title is not None:
            self.label = QtWidgets.QLabel(title)
            self.label.setObjectName('title')
            grid.addWidget(self.label, 0, 0, 1, 2)
            row = 1
        else:
            self.label = None
            row = 0
        
        # Side panel eg plot options
        if side is not None:
            grid.addWidget(side, row, 0)
            cols = 1
        else:
            cols = 2

        frame = QtWidgets.QFrame()
        grid.addWidget(frame, row, 2-cols, 1, cols)
        
        grid.setRowStretch(row, 1)
        grid.setColumnStretch(1, 1)
        
        self.grid = QtWidgets.QGridLayout(frame)
        
    
    def ChangeTitle(self, title):
        if self.label is None:
            return
        
        self.label.setText(title)
        
        
class LabelBox(QtWidgets.QFrame):
    """ A frame containing a value on top and its label below. Used for
    showing information
    
    Like a box frame but only showing text underneath
    """
    def __init__(self, label, value, *args, **kwargs):
        QtWidgets.QFrame.__init__(self, *args, **kwargs)

        # Adds border lines around the main widgets
        self.setObjectName('boxFrame')
        
        grid = QtWidgets.QGridLayout(self)
        
        # Adds a border to windows
        grid.setContentsMargins(1,1,1,1)
        grid.setSpacing(1)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(1, 1)
        
        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName('title')
        grid.addWidget(self.label)
        
        self.value = QtWidgets.QLabel(value)
        self.value.setObjectName('title')
        grid.addWidget(self.value)


    def ChangeValue(self, value):
        self.value.setText(value)
        
        
    def ChangeLabel(self, label):
        self.label.setText(label)
        
        
class CenterWidget(QtWidgets.QWidget):
    """"
    Made to easily center widgets in its row. Doesn't do anything else
    """
    def __init__(self, widget):
        QtWidgets.QWidget.__init__(self)
        
        self.widget = widget
        
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0,0,0,0)
        grid.addWidget(widget, 1, 1)
        
        for i in [0, 2]:
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        