import matplotlib.pyplot as plt
import numpy as np

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
            winding = self.windings[key]
            slope = self.slopes[key]
            l, r = self.regressions[key]
            l = int(l)
            r = int(r)

            plt.plot(winding)
            plt.plot(np.arange(l), np.ones(l) * np.mean(winding[:l]), c = 'red')
            plt.plot(np.arange(l, r), np.mean(winding[l:r]) + (slope * (np.arange(l, r) - (l + r - 1) / 2)), c = 'green')

            if len(winding[r:]):
                plt.plot(np.arange(r, len(winding)), np.ones(len(winding) - r) * np.mean(winding[r:]), c = 'purple')

            plt.title('Piecewise linear regression on winding number graph')
            
            plt.axvline(l, linestyle = '--', c = 'red')
            plt.axvline(r, linestyle = '--', c = 'purple')
            
            plt.xlabel('Residue number')
            plt.ylabel('Winding number')

            if save:
                plt.savefig(os.path.join(directory, key + '.pdf'))
                plt.close()
            else:
                plt.show()


# #plot regression with option to save file. return standard deviation of middle line segment
# def plot_regression(winding, params, slope, save = False, filename = ''):
#     l, r = params
#     l = int(l)
#     r = int(r)
#     pre, mid, post = get_premidpost(winding, params, slope)
    
#     plt.plot(winding)
#     plt.plot(np.arange(l), np.mean(winding[:l]), c = 'red')
#     plt.plot(np.arange(l, r), np.mean(winding[l:r]) + (slope * (np.arange(l, r) - (l + r - 1) / 2)), c = 'green')
#     plt.plot(np.arange(r, len(winding)), np.mean(winding[r:]), c = 'purple')

#     plt.title('Piecewise linear regression on winding number graph')
    
#     plt.axvline(l, linestyle = '--', c= 'r')
#     plt.axvline(r, linestyle ='--', c = 'purple')
    
#     plt.xlabel('Residue number')
#     plt.ylabel('Winding number')
    
#     if save:
#         plt.savefig(filename + '.png')
#         plt.close()
#     else:
#         plt.show()
#     return np.std(mid)

# def get_segs(winding, params, slope):
#     segs = []
    
#     breakpts = [0]+list(params)+[len(winding)]
#     for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):
#         a = int(a)
#         b = int(b)
#         seg = np.array(winding[a:b])
#         if ii%2:
#             try:
#                 seg -= slope*np.arange(a, b)
#             except:
#                 print(a, b, params, slope)
#                 print(seg)
#                 print((slope*np.arange(a, b)))
#                 raise Exception()
#         seg -= np.mean(seg)
#         segs.append(seg)

#     return segs

# #loss function for 4-breakpoint regression
# def loss_multi(winding, params, slope, penalties):
#     segs = get_segs(winding, params,slope)
#     return np.sum([penalties[ii%2]*np.sum(seg**2) for ii,seg in enumerate(segs)])

# # regression with 4 breakpoints
# def multi_regression(preX, l, r):
#     winding, s, c, q, dx = get_winding(preX)

#     m, scores = median_slope(winding, 150, 250)
#     pre, mid, post = get_premidpost(winding, (l, r), m)

    
#     start = np.where(np.diff(np.sign(pre)))[0][-1]
#     if preX.shape[0] - r:
#         end = r+np.where(np.diff(np.sign(post)))[0][0]
#     else:
#         end = preX.shape[0]
#     m, scores = median_slope(winding[start:end], 20, 30)

#     n = len(winding)
#     l = n // 2
#     r = (3 * n) // 4
#     parameters = np.array([l,l+(r-l)/3,l+2*(r-l)/3, r ])

#     penalties = [1, 1.5]
#     epsilon = 0.01
#     gradient = np.zeros(4)
#     delta = [*np.identity(4)]
#     prev_grad = np.array(gradient)
#     thresh = .3

#     for i in range(10000):
#         present = loss_multi(winding, parameters, m, penalties)
#         if np.linalg.norm(gradient - prev_grad)< thresh and i > 0:
#             break
#         gradient = np.array([loss_multi(winding, parameters + d, m, penalties) - present for d in delta])
#         parameters = parameters - epsilon * gradient
#     return winding, m, parameters

# #plot regression with 4 breakpoints
# def plot_regression_multi(winding, params, slope, save = False, filename = ''):
#     segs = get_segs(winding, params, slope)

#     plt.plot(winding)
#     breakpts = [0]+list(params)+[len(winding)]
#     for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):    
#         a = int(a)
#         b = int(b)        
#         g = winding[a:b]
#         plt.plot(range(a, b), g - segs[ii])
#     if save:
#         plt.savefig(filename + '.png')
#         plt.close()
#     else:
#         plt.show()
