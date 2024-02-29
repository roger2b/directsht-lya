import matplotlib.pyplot as plt
# make plots prettier
from matplotlib.pyplot import rc
import matplotlib.font_manager

rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
rc('text', usetex=True)
rc('legend',**{'fontsize':'18'})

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = '18'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 1.3
plt.rcParams['ytick.labelsize'] = '18'
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 1.3
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.labelsize'] = '18'
plt.rcParams['axes.autolimit_mode'] = 'data'
plt.rcParams['legend.fontsize'] = '18'
plt.rcParams['font.size'] = '18'
plt.rcParams['figure.titlesize'] = '18'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['figure.figsize'] = 12,8


plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 25 
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'