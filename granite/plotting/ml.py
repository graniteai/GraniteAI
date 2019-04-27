# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:20:55 2018

@author: Mike Staddon
"""

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, auc

from ..ml import stats

from .colors import colors

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
    ax.grid(False)
    
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
        ax.plot(fpr, tpr, color=colors[index],
                lw=3, label='{:} (Area = {:.3f})'.format(cat, roc_auc))
    #    ax.fill_between(fpr, tpr, color='C0', alpha=0.33)
    
    ax.plot([0, 1], [0, 1], color=(0.85, 0.85, 0.85), lw=3, linestyle='--')

    ax.legend(fontsize=10,
              frameon=False,
              loc='center left',
              bbox_to_anchor=(1.05, 0.5))
    
#    FormatChart(fig, ax, xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    ax.set_aspect('equal')
    
    #Background
    ax.patch.set_alpha(1)
    
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
        
        ax.scatter(time, scores, color=colors[1], label='Run Score', zorder=1)
        ax.plot(min_time, min_loss, lw=3, color=colors[0], label='Best Score', zorder=2,
                marker='o')
    
        # Manually make legend
        leg = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', 
                    frameon=False, markerscale=0, handlelength=0)
        
        i = 0
        for text in leg.get_texts():
            text.set_color([colors[0], colors[1]][i])
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
    
    fig.tight_layout()
    
    
def FeatureDependence(X, y, feature, dtype,
                      bins=None, model=None, cat=None, target=None,
                      ylims=None,
                      fig=None, ax=None):
    """ Plot how the mean value or proportion of y changes with feature 
    TO DO: rename as x and y in to be consistent
    
    Arguments:
        
    """
    
    means, counts, bins = stats.feature_dependence(X, y, feature, dtype, model=model, cat=cat, bins=bins)
    
    if target is None:
        target = ''
        
        if cat is None:
            ylabel = 'Average ' + target
        else:
            ylabel = 'Proportion ' + target + ': ' + str(cat)
        
    if dtype == 'numeric':
        if fig is None:
            fig = plt.figure()
            
        if ax is None:
            ax = fig.add_subplot(111)

        # Histogram of counts
        width = bins[1] - bins[0]
        ax.bar(bins, counts, width=width, color=colors[0],
               edgecolor=(0.85, 0.85, 0.85), lw=0.5, zorder=10)
        
        # Average line plots
        ax2 = ax.twinx()
        
        bins, means = bins[counts != 0], means[counts != 0]
        
        ax2.plot(bins, means['y'], color=colors[1], lw=3,
                 marker='x', ms=9, mew=3)
        
        if model is not None:
            ax2.plot(bins, means['ypred'], color=colors[2], lw=3, 
                    marker='+', ms=9, mew=3)
            
            
        # A dummy axis to draw grid lines underneath!        
        # Third axis is just for grid lines, can't draw averages over counts over lines!

        ax.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        
        ax2.set_yticks([])
        ax2.set_ylim(ylims)
        ax2.set_ylabel(ylabel)
        ax2.grid(False)
        

        # Make legend
        handles = [matplotlib.lines.Line2D([0], [0], lw=3,
                                           color=[colors[1], colors[2]][j],
                                           label=['Actual', 'Predicted'][j]) for j in range(2)]
    
        ax2.legend(handles=handles, frameon=False,
                   loc='upper left', bbox_to_anchor=(1, 1))
        

        fig.tight_layout()
        
