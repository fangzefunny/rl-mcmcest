import numpy as np 
import seaborn as sns 
import matplotlib.colors
import matplotlib as mpl 

class viz:
    '''Define the default visualize configure
    '''
    # basic
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    lBlue2  = np.array([166, 201, 222]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    lGreen  = np.array([242, 251, 238]) / 255
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255
    lRed2   = np.array([254, 177, 175]) / 255
    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    ocGreen = np.array([ 90, 196, 164]) / 255
    oGrey   = np.array([176, 166, 183]) / 255
    Palette = [Blue, Yellow, Red, ocGreen, Purple]

    # palette for agents
    b1      = np.array([ 43, 126, 164]) / 255
    r1      = np.array([249, 199,  79]) / 255
    r2      = np.array([228, 149,  92]) / 255
    r3      = np.array([206,  98, 105]) / 255
    m2      = np.array([188, 162, 149]) / 255
    g       = np.array([.7, .7, .7])
    Pal_agent = [b1, g, Red, r2, m2] 

    # palette for block types
    dGreen  = np.array([ 15,  93,  81]) / 255
    fsGreen = np.array([ 79, 157, 105]) / 255
    Ercu    = np.array([190, 176, 137]) / 255
    Pal_type = [dGreen, fsGreen, Ercu]

    # Morandi
    m0      = np.array([101, 101, 101]) / 255
    m1      = np.array([171,  84,  90]) / 255
    m2      = np.array([188, 162, 149]) / 255
    Pal_fea = [m0, m1, m2]

    # for insights
    BluesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue, Blue])
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed, dRed])
    YellowsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow, Yellow])
    GreensMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizGreens',  [lGreen, Green])

    @staticmethod
    def get_style(): 
        # Larger scale for plots in notebooks
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        mpl.rcParams['axes.spines.right']  = False
        mpl.rcParams['axes.spines.top']    = False
        mpl.rcParams['font.weight']        = 'regular'
        mpl.rcParams['savefig.format']     = 'pdf'
        mpl.rcParams['savefig.dpi']        = 300
        mpl.rcParams['figure.facecolor']   = 'w'
        mpl.rcParams['figure.edgecolor']   = 'None'
        mpl.rcParams['axes.facecolor']     = 'None'
        mpl.rcParams['legend.frameon']     = False
        mpl.rcParams['axes.spines.right']  = False
        mpl.rcParams['axes.spines.top']    = False
