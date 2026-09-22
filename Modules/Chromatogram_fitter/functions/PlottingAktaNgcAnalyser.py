import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Small colorblind-friendly qualitative palette (Okabe-Ito), used instead of
# an extra plotting-palette dependency for the handful of colors we need.
_PALETTE = [
    '#0072B2',  # blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#D55E00',  # vermillion
    '#CC79A7',  # pink
]

def get_fraction_volumes(df):
    start_volumes = df.groupby('Fraction')['Volume [mL]'].first()
    end_volumes = df.groupby('Fraction')['Volume [mL]'].last()

    fraction_volumes = np.array([start_volumes.index, start_volumes.values, end_volumes.values]).T

    fraction_volumes = fraction_volumes[fraction_volumes[:,1].argsort()]

    return fraction_volumes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def linePlot3axis(Data):
    x = Data['Volume [mL]']#place UV
    
    columns = ['UV [mAU]', 'B [%B]', 'Cond [mS/cm]']
    common_columns = [col for col in columns if col in Data.columns] #=list(set(Data.columns) & set(columns))
    N = len(common_columns)
    
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(10,6) #width,height
    gs = gridspec.GridSpec (6, 2,width_ratios=[3,2],height_ratios=[1,1,1,1,1,1])
    ax1 = fig.add_subplot(gs[0:6, 0])

    axes = [ax1, ax1.twinx(), ax1.twinx()]
    axes[-1].spines['right'].set_position(('axes', 1.2))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    colors = _PALETTE[:3]
    fig.suptitle(Data['Name'].iloc[0])
    
    i = 0
    for axesi, color, label in zip(axes, colors, common_columns):
        y = Data[label]
        axesi.plot(x,y, color=color)
        axesi.tick_params(axis='y', colors=color)
        axesi.set_ylabel(label,color=color)
        axi = fig.add_subplot(gs[int(i*6/N):int((i+1)*6/N), 1])
        axi.plot(x,y,color=color)
        axi.set_title(label)
        axi.set_ylabel(label[label.rfind('['):])
        axi.set_xlabel('[mL]')
        i=i+1
    axes[0].set_xlabel('[mL]')

def linePlotProcess(Data1, Data2, Yaxis, title=None, label1='After', label2='Before'):
    """Overlay two versions of the same signal, e.g. before/after a processing step.

    Data1 is drawn as a solid coloured line (on top), Data2 as a dashed black
    line underneath, so the effect of a processing step is easy to see.
    Falls back to a generic colour for any column name that isn't one of the
    three "known" chromatogram signals, so this also works for e.g.
    '[Elutant] [mM]'.
    """
    x1 = Data1['Volume [mL]']
    x2 = Data2['Volume [mL]']

    y1 = Data1[Yaxis]
    y2 = Data2[Yaxis]

    color_map = {
        'UV [mAU]': _PALETTE[0],
        'B [%B]': _PALETTE[1],
        'Cond [mS/cm]': _PALETTE[2],
        '[Elutant] [mM]': _PALETTE[3],
    }
    color = color_map.get(Yaxis, _PALETTE[4])

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6,4.5) #width,height
    gs = gridspec.GridSpec (1, 1,width_ratios=[1],height_ratios=[1])
    ax = fig.add_subplot(gs[0, 0])

    ax.plot(x2,y2,color='k',linestyle='dashed', label=label2)
    ax.plot(x1,y1,color=color,linewidth=2, label=label1)
    ax.set_xlabel('Volume [mL]')
    ax.set_ylabel(Yaxis)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    return fig

def linePlotFractions(Data,Config):
    if Config['System']=='SystemA' or Config['System']=='SystemB':
        N=1
        f =  get_fraction_volumes(Data)
        
        fig = plt.figure(tight_layout=True)
        fig.set_size_inches(N*7,4.5) #width,height
        gs = gridspec.GridSpec (1, N)
        
        nfrac = f[1:,0]
        vfrac = f[1:,1]
        
        for i in range(N):
            ax = fig.add_subplot(gs[0, i])
            Nend = np.where(nfrac=='Waste')[0][0]
            yLim = 100+Data[(Data['Volume [mL]'] >= vfrac[0]) & (Data['Volume [mL]'] <= vfrac[Nend])].max()['UV [mAU]']
            #yLim = 300
    
            if i !=1:
                for j in range(Nend):
                    ax.text(vfrac[j],yLim,nfrac[j]) 
    
            for j in range(Nend):
                ax.vlines(vfrac[j],0,yLim)
                if ((j+1) % 5 )==0:
                    ax.text(vfrac[j],yLim,nfrac[j])
        ax.plot(Data['Volume [mL]'],Data['UV [mAU]'],'c')
        ax.set_title('UV')
        ax.set_ylabel('mAU')
        ax.set_xlabel('mL')
        ax.set_xlim([vfrac[0]-5,vfrac[Nend]])
        ax.set_ylim([0,yLim+10])
        nfrac=nfrac[Nend+1:]
        vfrac=vfrac[Nend+1:]
        fig.suptitle(Data['Name'].iloc[0])
    else:
        print('I have not yet developed code for plotting the fractions using NGC data')

def addFractions(Ax,Fraction,LogShift=0):
    def postSlash(input_string):
        match = re.search(r'/(\d+)$', input_string)
        if match:
            return match.group(1)
        return None
    End = Fraction.index[-1]
    limY = Ax.get_ylim()
    limX = Ax.get_xlim()
    for j in range(End):
        if Fraction['End(CV)'].iloc[j]-LogShift>limX[0]:
            label = postSlash(Fraction['Rack/Tube'].iloc[j])
            Ax.vlines(Fraction['End(CV)'].iloc[j]-LogShift,0,limY[1],'0.8')
            Ax.text((Fraction['Start(CV)'].iloc[j]+Fraction['End(CV)'].iloc[j]-2*LogShift)/2,
                    limY[1]*1.08,
                    label,
                    fontsize=5)
    Ax.set_xlim(Fraction['Start(CV)'].iloc[0],Fraction['End(CV)'].iloc[End])
    Ax.set_xlim(limX)