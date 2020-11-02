import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Courier New'
    mpl.rcParams['text.hinting'] = False
    # Set colors cycle
    colors = mpl.cycler('color', ['#3388BB', '#EE6666', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    #plt.rc('figure', facecolor='#00000000', edgecolor='black')
    #plt.rc('axes', facecolor='#FFFFFF88', edgecolor='white', axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')

# define function that allows to generate a number of sub plots in a single line with the given titles
def prep_plots(titles, fig_size, fig_num=1):
    """
    create a figure with the number of sub_plots given by the number of totles, and return all generated subplot axis
    as a list
    """
    # first close possibly existing old figures, if you dont' do this Juyter Lab will coplain after a while when it collects more than 20 existing ficgires for the same cell
    # plt.close(fig_num)
    # create a new figure
    hdr_plot_style()
    fig=plt.figure(fig_num, figsize=fig_size)
    ax_list = []
    for ind, title in enumerate(titles, start=1):
        ax=fig.add_subplot(1, len(titles), ind)
        ax.set_title(title)
        ax_list.append(ax)
    return ax_list
    
def finalize_plots(axes_list, legend=True, fig_title=None):
    """
    adds grid and legend to all axes in the given list
    """
    if fig_title:
        fig = axes_list[0].figure
        fig.suptitle(fig_title, y=1)
    for ax in axes_list:
        ax.grid(True)
        if legend:
            ax.legend()

def plot_patterns(P,D):
    """ Plots the decision boundary of a single neuron with 2-dimensional inputs """
    hdr_plot_style()
    nPats = P.shape[1]
    nUnits = D.shape[0]
    if nUnits < 2:
        D = np.concatenate(D, np.zeros(1,nPats)) 
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    # Calculate the bounds for the plot and cause axes to be drawn.
    xmin, xmax = np.min(P[0, :]), np.max(P[0, :])
    xb = (xmax - xmin) * 0.2
    ymin, ymax = np.min(P[1, :]), np.max(P[1, :])
    yb = (ymax-ymin) * 0.2
    ax.set(xlim=[xmin-xb, xmax+xb], ylim=[ymin-yb, ymax+yb])
    plt.title('Input Classification')
    plt.xlabel('x1'); plt.ylabel('x2')
    # classVal = 1 + D[1,:] + 2 * D[2,:];
    colors = [[0, 0.2, 0.9], [0, 0.9, 0.2], [0, 0, 1], [0, 1, 0]]
    symbols = 'ooo*+x'; Dcopy = D[:]
    #Dcopy[Dcopy == 0] = 1
    for i in range(nPats):
        c = Dcopy[i]
        ax.scatter(P[0,i], P[1,i], marker=symbols[c], c=colors[c], s=50, linewidths=2, edgecolor='w')
    #ax.legend()
    ax.grid(True)
    return fig

def plot_boundary(W,iVal,style,fig):
    """ Plots (bi-dimensionnal) input patterns """
    nUnits = W.shape[0]
    colors = plt.cm.inferno_r.colors[1::3]
    xLims = plt.gca().get_xlim()
    for i in range(nUnits):
        if len(style) == 1:
            color = [1, 1, 1];
        else:
            color = colors[int((3 * iVal + 9) % len(colors))]
        plt.plot(xLims,(-np.dot(W[i, 1], xLims) - W[i, 0]) / W[i, 2], linestyle=style, color=color, linewidth=1.5);
        fig.canvas.draw()
        
def visualize_boundary_linear(X, y, model):
# VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
#   learned by the SVM and overlays the data on it
    hdr_plot_style()
    w = model["w"]
    b = model["b"]
    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).transpose()
    yp = - (w[0] * xp + b) / w[1]
    plt.figure(figsize=(12, 8))
    pos = (y == 1)[:, 0] 
    neg = (y == -1)[:, 0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', linewidths=2, s=23, c=[0, 0.5, 0])
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', linewidths=2, s=23, c=[1, 0, 0])
    plt.plot(xp, yp, '-b')
    plt.scatter(model["X"][:, 0], model["X"][:, 1], marker='o', linewidths=4, s=40, c=None, edgecolors=[0.1, 0.1, 0.1])
    
def plot_data(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0
    hdr_plot_style()
    # Find Indices of Positive and Negative Examples
    pos = (y == 1)[:, 0] 
    neg = (y == 0)[:, 0]
    # Plot Examples
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', edgecolor='k', linewidths=2, s=50, c=[0, 0.5, 0])
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', edgecolor='k', linewidths=2, s=50, c=[1, 0, 0])
    return fig

def visualize_boundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it
    hdr_plot_style()
    # Plot the training data on top of the boundary
    plot_data(X, y)
    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).transpose()
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).transpose()
    [X1, X2] = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.vstack((X1[:, i], X2[:, i]))
        vals[:, i] = svmPredict(model, this_X)
    # Plot the SVM boundary
    plt.contour(X1, X2, vals, [1, 1], c='b')
    # Plot the support vectors
    plt.scatter(model["X"][:, 0], model["X"][:, 1], marker='o', linewidths=4, s=10, c=[0.1, 0.1, 0.1])
    
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='w',
               levels=[-1, 0, 1], alpha=0.9,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=2, edgecolor='w', facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)