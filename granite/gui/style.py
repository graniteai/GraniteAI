# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 20:35:59 2018

@author: Mike Staddon

Defines the style of the different widgets
"""

def max_width():
    return 1080


def max_height():
    return 1080


def colors(i):
    return ['#FF882B', '#3B68B9'][i % 2]


def app_css():
    """ Returns the css for the Granite AI app """
    
    # Shades
    darkest = '#202020'
    darker = '#303030'
    dark = '#606060'
    medium = '#A0A0A0'
    light = '#C0C0C0'
    lighter = '#F0F0F0'
    lightest = '#FFFFFF'
    
    # Blue - and highlights
    color_1 = '#3B68B9'
    color_1_light = '#5B88D9'
    color_1_lightest = '#8BA8F9'
    
    # Orange
    color_2 = '#FF882B'
    
    # Purple
    color_3 = '#F45B69'
    
    # Define colors - prefix means a theme
    colors = {'font_fg': darkest,
              'warning_fg': color_1,
              'top_bg': darkest,
              'top_fg': lightest,
              'default_bg': lightest,
              'highlight_1': color_1,
              'highlight_2': color_2,
              'button_bg': color_1,
              'button_bg_highlight': color_1_light,
              'button_fg': lightest,
              'button_bg_disabled': color_1_lightest,
              'subtle_button_bg': medium,
              'subtle_button_fg': lightest,
              'subtle_button_bg_highlight': light,
              'style_line_bg': medium,
              'bg': lightest,
              'color_1': color_1,
              'color_2': color_2,
              'color_3': color_3,
              'background_bg': lighter,
              'light': light}
    
    colors = {'darkest': darkest,
              'darker': darker,
              'dark': dark,
              'medium': medium,
              'light': light,
              'lighter': lighter,
              'lightest': lightest,
              'color_1': color_1,
              'color_1_light': color_1_light,
              'color_1_lightest': color_1_lightest,
              'color_2': color_2,
              'color_3': color_3}
    
    
    # Use double braces for real braces because of .format()
    
    #Default frames
    css = """
    GraniteApp
    {{
    background: {darkest};
    }}
    
    QFrame#background
    {{
    background: {lighter};
    border: 0px;
    }}
    
    QWidget#color_1
    {{
    background: {color_1};
    }}
    
    QWidget#color_2
    {{
    background: {color_2};
    }}
    
    QWidget#color_3
    {{
    background: {color_3};
    }}

    QWidget
    {{
    font: 14px;
    font-family: arial;
    }}
    
    QLabel
    {{
    color: {darker};
    }}
    
    QLabel#warning
    {{
    color: {color_1};
    }}
    
    QFrame
    {{
    background: {lightest};
    }}
    
    StyleLine
    {{
    background: {medium};
    }}
    
    QPushButton
    {{
    background: {color_1};
    color: {lightest};
    border: 0px;
    border-radius: 5px;
    padding: 8px;
    font: bold;
    }}
    
    QPushButton::hover
    {{
    background: {color_1_light};
    }}
    
    QPushButton::disabled
    {{
    background: {color_1_lightest};
    }}
    
    QPushButton#subtle
    {{
    background: {medium};
    color: {lightest};
    }}
    
    QPushButton#subtle::hover
    {{
    background: {light};
    }}
    """.format(**colors)
    
    # Menu bar
    css += """
    QFrame#saveBar
    {{
    background: {darker};
    }}
    
    QPushButton#saveBar
    {{
    color: {lightest};
    background: {darker};
    border: 0px;
    font: bold 16px;
    padding: 0px;
    }}
    
    QPushButton#saveBar::hover
    {{
    color: {color_2};
    }}""".format(**colors)
    
    # Nav bar
    css += """
    
    QFrame#navBar
    {{
    background: {dark};
    }}
    
    QLabel#navBar
    {{
    background: {dark};
    color: {lightest};
    border: 0px;
    font: bold 16px;
    padding: 0px;
    }}
    
    QPushButton#navBar
    {{
    background: {dark};
    color: {lightest};
    border: 0px;
    font: bold 16px;
    padding: 0px;
    }}
    
    QPushButton#navBar::hover
    {{
    color: {color_2};
    }}
    """.format(**colors)
    
    # Regular tabs and tab widgets
    css += """
    QTabBar
    {{
        background: {lighter};
        font: 14px;
        border: 2px solid {lighter};
        border-bottom-color: {lighter};
        color: {dark};
        min-width: 24ex;
        min-height: 8ex;
    }}
    
    
    QTabBar::tab
    {{
        background: {lighter};
        font: 14px;
        border: 2px solid {lighter};
        border-bottom-color: {lighter};
        color: {dark};
        min-width: 24ex;
        min-height: 8ex;
    }}
    
    
    QTabBar::tab:selected
    {{
        color: {darkest};
        border-color: {lighter};
        border-bottom-color: {color_1};
    }}
    
    QTabBar::tab:disabled
    {{
        width: 0; height: 0; margin: 0; padding: 0; border: none;
        min-width: 0px;
        min-height: 0px;
    }}
    
    QTabWidget
    {{
          background: {lighter};
    }}
    
    QTabWidget::pane
    {{
        border: 0px;
        background: {lighter};
    }}
    """.format(**colors)
    
    # Horizontal tabs
    css += """
    HorizontalTabBar::tab
    {{
        background: {darkest};
        border: 4px solid {darkest};
        border-right-color: {darkest};
        color: {lightest};
    }}
    
    
    HorizontalTabBar::tab:selected
    {{
        background: {darker};
        color: {lightest};
        border-color: {darker};
        border-right-color: {color_2};
    }}
    """.format(**colors)
    
    # Tables
    css += """
    QHeaderView::section
    {{
        background-color: {lightest};
        padding-top: 8px;
        padding-bottom: 8px;
        padding-right: 16px;
        color: {darker};
        border-left: 0px;
        border-right: 0px;
        border-top: 0px;
        border-bottom: 1px solid {medium};
    }}
    
    QTableView
    {{
         background: {lightest};
         alternate-background-color: {lightest};
         border: 0px;
    }}

    QPushButton#table
    {{
        background: {lightest};
        border: 0px;
        text-align: left;
        color: {color_1};
    }}
    
    QPushButton#table::hover
    {{
        color: {color_2};
    }}
    """.format(**colors)
    
    # Boxes
    css += """
    QFrame#boxFrame
    {{
        background: {light};
    }}
    
    QLabel#title
    {{
        font: 18px;
        padding: 8px;
    }}
    
    QLabel#subtitle
    {{
        font: 16px;
    }}
    """.format(**colors)

    return css

