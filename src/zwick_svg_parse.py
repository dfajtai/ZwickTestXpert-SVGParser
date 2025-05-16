import os, sys, glob, re

from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from svgelements import SVG, Path, Text, Line

def find_longest_one_sequence_indices(arr, n=1):
    """Returns start/end indices of the first n longest 1s sequences using zero-crossing logic."""
    arr = np.asarray(arr)
    if len(arr) == 0:
        return []

    # Detect transitions using forward difference
    diff = np.diff(arr)
    starts = np.where(diff == 1)[0] + 1  # 0→1 transitions (+1 for original index)
    ends = np.where(diff == -1)[0]       # 1→0 transitions (original index)

    # Handle array boundaries
    if arr[0] == 1:
        starts = np.insert(starts, 0, 0)
    if arr[-1] == 1:
        ends = np.append(ends, len(arr ) -1)

    # Create sequence candidates
    sequences = []
    for s, e in zip(starts, ends):
        if s <= e:  # Validate sequence
            sequences.append((s, e, e - s + 1))

    # Sort by length (descending) then position (ascending)
    sequences.sort(key=lambda x: (-x[2], x[0]))

    return [(s, e) for s, e, _ in sequences[:n]]



def parse_svg_plot(svg_path:str, min_segment_len:int = 100, 
                   x_min:float = 0, y_min:float = 0, 
                   x_max:float = 20, y_max:float =2500,
                   x_invert:bool = False, y_invert:bool = False)-> List[pd.DataFrame]:
    """
    Parse SVG plot and return a DataFrame with the data.
    """
    # Read the SVG file
    svg_data = SVG.parse(svg_path)
    
    assert len(svg_data) >= 0, "SVG file should contain at least one element."
    
    # Extract the data from the SVG elements
    plot_elements = svg_data[-1]
    
    texts = []
    markers = []
    datasets = []
    
    for element in plot_elements[0]:
        if isinstance(element, Text):
            texts.append(element)

        elif isinstance(element, Path):
            segments = element.segments()[1:]
            if len(segments) ==1:
                markers.append(segments[0])
            elif len(segments)>=min_segment_len:
                _points = []
                                
                for i, _segment in enumerate(segments):
                    if not isinstance(_segment,Line):
                        continue
                    if i==0:
                        _points.append([_segment.start.x,_segment.start.y])
                        _points.append([_segment.end.x,_segment.end.y])
                    else:
                        _points.append([_segment.end.x,_segment.end.y])
                datasets.append(_points)
    
    
    x_axis = None
    marker_x_max_len = 0
    y_axis = None
    marker_y_max_len = 0
    
    for m in markers:
        x_len = abs(m.end.x - m.start.x)
        y_len = abs(m.end.y - m.start.y)
        
        if x_len > marker_x_max_len:
            marker_x_max_len = x_len
            x_axis = m
        
        if y_len > marker_y_max_len:
            marker_y_max_len = y_len
            y_axis = m    
    
    assert x_axis.start.x == y_axis.start.x and x_axis.start.y == y_axis.start.y, "Axes should start from the same point"
    
    # calculate plot area origin
    origin = np.array([x_axis.start.x,x_axis.start.y])
    
    # calculate plot borders
    x_lim = x_axis.end.x
    y_lim = y_axis.end.y
    
    # calculate plot size
    x_size = x_lim - origin[0]
    y_size = y_lim - origin[1]
    
    # scaling factor to map svg to data scale
    x_factor = (x_max-x_min) / x_size
    y_factor = (y_max-y_min) / y_size
    
    df_list = []
    
    for dataset in datasets:
        data = np.array(dataset)
        
        # displacement
        data = data - origin
        
        x = data[:,0]*x_factor
        y = data[:,1]*y_factor
              
    
        # invert
        if x_invert:
            x = x_max-x
        if y_invert:
            y = y_max-y

        df = pd.DataFrame({"x":list(x),"y":list(y)})
        
        df_list.append(df)

    return df_list



if __name__ == "__main__":
    # Example usage
    svg_path = "sample/20_2500_20_1x1.svg"
    df_list = parse_svg_plot(svg_path)
    
    
    for i, df in enumerate(df_list):
        if not isinstance(df,pd.DataFrame):
            continue
        
        df.to_csv(f"sample/curve_{str(i+1).zfill(2)}.csv",index = False, float_format="%.3f")
        
        
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'])
        ax.set_xlabel('travel in mm')
        ax.set_ylabel('force in N')
        
        
        # Y-axis: major grid every 500, minor grid every 100
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        
        ax.set_xlim(0,20)
        ax.set_ylim(0,2500)
        
        ax.grid(True, axis='y', which='major', linestyle=':', linewidth=0.5, color='black')
        ax.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.2, color='gray')

        # X-axis: major grid every 5, minor grid every 1
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, axis='x', which='major', linestyle=':', linewidth=0.5, color='black')
        ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.2, color='gray')
        
        
        plt.savefig(f"sample/curve_{str(i+1).zfill(2)}.png")  # Save the figure
        plt.close()
        # plt.show()



