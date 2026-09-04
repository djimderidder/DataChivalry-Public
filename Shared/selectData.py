import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.widgets as mwidgets
import ipywidgets as widgets
import re
import numpy as np
import pandas as pd
import time
from IPython.display import display

class Well96Selector:
    def __init__(self, Data):
        self.well_dict = Data.dict
        self.well_data = Data.data
        self.selected_wells = []
        self.baseline0_well = []
        self.baseline1_well = []
        self.fig = plt.figure(tight_layout=True)
        self.fig.set_size_inches(9.6,4) #width,height
        gs2 = gridspec.GridSpec (1, 2,width_ratios=[5.6,4],height_ratios=[1])
        self.ax1 = self.fig.add_subplot(gs2[0,0])        
        self.ax2 = self.fig.add_subplot(gs2[0,1])
        self.text = None
        self._create_plot()
        self._create_widgets()

    def _create_plot(self):
        wells = list(self.well_dict.keys())
        x = [int(re.findall(r'\d+', well)[0]) for well in wells]
        y = [ord([l for l in well if l.isupper()][0]) - ord('A') + 1 for well in wells]
        self.scatter = self.ax1.scatter(x, y, picker=True)
        self.ax1.set_xticks(range(1, 13))
        self.ax1.set_xticklabels(range(1, 13))
        self.ax1.set_yticks(range(1, 9))
        self.ax1.set_yticklabels([chr(i + ord('A') - 1) for i in range(1, 9)])
        self.ax1.set_aspect('equal')
        self.ax1.invert_yaxis()

        # Set frame dimensions
        xleft = 1 - 14.38 / 9
        xright = 12 + 14.38 / 9
        ytop = 1 - (4.5 + 7.01) / 9
        ybottom = 8 + (4.5 + 7.01) / 9
        self.ax1.set_xlim(xleft - 1, xright + 1)
        self.ax1.set_ylim(ybottom + 1, ytop - 1)

        # Draw wells
        radius = 4 / 9
        for i in range(1, 13):
            for j in range(1, 9):
                circle = patches.Circle((i, j), radius, edgecolor='black', facecolor='none', linewidth=1)
                self.ax1.add_patch(circle)

        # Draw frame with cut-out corner
        xc1 = xleft + np.sqrt(2 * 7.85) / 9
        yc1 = ytop
        xc2 = xleft
        yc2 = ytop + np.sqrt(2 * 7.85) / 9

        self.ax1.plot([xc1, xright], [ytop, ytop], color='black', linewidth=1)
        self.ax1.plot([xleft, xleft], [yc2, ybottom], color='black', linewidth=1)
        self.ax1.plot([xc1, xc2], [yc1, yc2], color='black', linewidth=1)
        self.ax1.plot([xright, xright], [ytop, ybottom], color='black', linewidth=1)
        self.ax1.plot([xleft, xright], [ybottom, ybottom], color='black', linewidth=1)

        self.cid = self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        plt.show()

    def _create_widgets(self):
        self.select_button = widgets.Button(description="Sort Selection")
        display(self.select_button)
        self.select_button.on_click(self._on_select_button_click)

    def _on_pick(self, event):
        self.fig.canvas.mpl_disconnect(self.cid)  # Disconnect event handler
        ind = event.ind[0]
        well = list(self.well_dict.keys())[ind]
        if well in self.selected_wells:
            self.selected_wells.remove(well)
            if self.baseline0_well == []:
                self.baseline0_well.append(well)
            elif self.baseline1_well == []:
                self.baseline1_well.append(well)
        elif well in self.baseline0_well:
            self.baseline0_well = []
            if self.baseline1_well == []:
                self.baseline1_well.append(well)
        elif well in self.baseline1_well:
            self.baseline1_well = []
        else:
            self.selected_wells.append(well)
        self._update_colors()
        self._display_well_content(well)
        
        data = self.well_data[list(self.well_dict.keys())[ind]]
        
        self.ax2.cla()  # Clear the current axis
        for i in self.well_data['Gain'].unique():
            gaini = self.well_data['Gain']==i
            datai = data[gaini]
            t = self.well_data[gaini].iloc[:, 1]
            self.ax2.plot(t, datai)
        self.ax2.set_xlabel(self.well_data.columns[1])
        self.ax2.set_ylim([0,self.well_data.max()[2:].max()])
        
        time.sleep(0.3)  # Add a small delay
        self.cid = self.fig.canvas.mpl_connect('pick_event', self._on_pick)  # Reconnect event handler

    def _update_colors(self):
        colors = ['royalblue' if well in self.selected_wells else 'lightblue' if well in self.baseline0_well else 'darkblue' if well in self.baseline1_well else 'gray' for well in self.well_dict.keys()]
        self.scatter.set_facecolor(colors)
        self.fig.canvas.draw()

    def _display_well_content(self, well):
        if self.text:
            self.text.remove()
        ytop = 1 - (4.5 + 7.01) / 9
        self.text = self.ax1.text(6.5, ytop - 0.5, f"Well: {well}\nContent: {self.well_dict[well]}", 
                                 ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        self.fig.canvas.draw()

    def _on_select_button_click(self, b):
        self.selected_wells = sorted(self.selected_wells, key=lambda well: list(self.well_dict.keys()).index(well))
        print(f"Selected Wells: {self.selected_wells}, baseline: ({self.baseline0_well},{self.baseline1_well})")

    def roi(self):
        roi = pd.DataFrame({
            'well': self.selected_wells,
            'nFibril': [self.baseline1_well[0]] * len(self.selected_wells),
            'nBase': [self.baseline0_well[0]] * len(self.selected_wells)
        })
        return roi
 
