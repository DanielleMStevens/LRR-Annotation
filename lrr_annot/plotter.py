import matplotlib.pyplot as plt
import numpy as np
import os

def plot_regression(ax, winding, breakpoints, slope):
    """
    ax: matplotlib axis
        Axis on which to plot the regression
	winding: ndarray(n)
		The winding number at each residue
    breakpoints: ndarray(int)
        Residue locations of the breakpoints
    slope: ndarray(n)
			Estimated slope in each winding segment
    """
    boundaries = [0] + breakpoints.tolist() + [len(winding)]
    ax.plot(winding, c='C0', linewidth=1, zorder=100)
    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        linear = (i % 2) * slope * (np.arange(a, b) - (a + b - 1) / 2)
        y = linear + np.mean(winding[a:b])
        ax.plot(np.arange(a, b), y, c=f"C{i+1}", linestyle='--', linewidth=3)
    for b in breakpoints:
        ax.axvline(b, linestyle='--', c='k')
    ax.set_title('Piecewise linear regression on winding number graph')
    ax.set_xlabel('Residue number')
    ax.set_ylabel('Winding number')

class Plotter:
    def __init__(self):
        self.windings = {}
        self.regressions = {}
        self.slopes = {}

    def load(self, windings, regressions, slopes):
        self.windings.update(windings)
        self.regressions.update(regressions)
        self.slopes.update(slopes)

    def plot_regressions(self, save = False, directory = '', progress = True):
        from tqdm import tqdm
        for key in (tqdm(self.regressions, desc = 'Making plots') if (save and progress) else self.regressions):
            plt.clf()
            plot_regression(plt.gca(), self.regressions[key], self.windings[key], self.slope[key])
        if save:
            plt.savefig(os.path.join(directory, key + '.pdf'))
            plt.close()
        else:
            plt.show()
        