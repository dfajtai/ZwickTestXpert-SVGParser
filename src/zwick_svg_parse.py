import os, sys, glob, re

from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import scipy.signal
import scipy.stats
from svgelements import SVG, Path, Text, Line

import scipy
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter


def parse_svg_plot(svg_path:str, min_segment_len:int = 100, 
                   x_min:float = 0, y_min:float = 0, 
                   x_max:float = 20, y_max:float =2500,
                   x_invert:bool = False, y_invert:bool = False,
                   start_skip = 5, end_skip = 5)-> List[pd.DataFrame]:
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
        data = np.array(dataset)[start_skip:-end_skip,:]
                
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


def save_parsed_data(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]"):
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
        
    col_lookup = {"x":x_label,"y":y_label}

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {idx} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {idx} is invalid. (Missing column.)")
            continue
        
        _df = df.copy()
        _df.columns = [col_lookup.get(c) for c in df.columns]
        
        out_path = os.path.join(out_dir,f"{name_extension}_{str(idx).zfill(index_length)}.csv")
        _df.to_csv(out_path,index = False, float_format="%.3f")
        
        print(f"Sample {str(idx).zfill(index_length)} saved at {out_path}")
        

def save_parsed_data_as_plots(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     x_lim:list = [0,20], y_lim:list = [0,2500]):
    
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
        
    col_lookup = {"x":x_label,"y":y_label}

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {idx} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {idx} is invalid. (Missing column.)")
            continue
        
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'])
        ax.set_xlabel(col_lookup.get('x'))
        ax.set_ylabel(col_lookup.get('y'))
        
        ax.set_title(f'Sample {str(idx).zfill(index_length)}')
        
        # Y-axis: major grid every 500, minor grid every 100
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        
        ax.grid(True, axis='y', which='major', linestyle=':', linewidth=0.5, color='black')
        ax.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.2, color='gray')

        # X-axis: major grid every 5, minor grid every 1
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, axis='x', which='major', linestyle=':', linewidth=0.5, color='black')
        ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.2, color='gray')
        
        out_path = os.path.join(out_dir,f"{name_extension}_{str(idx).zfill(index_length)}.png")
        
        plt.savefig(out_path)  # Save the figure
        plt.close()
        
        print(f"Plot created for sample {str(idx).zfill(index_length)} at {out_path}")
        
        
def analyze_data(x:np.ndarray,y:np.ndarray, target_num_of_extremas:int=2, min_value_factor:float = 0.2)->dict:
    assert isinstance(x,np.ndarray) and isinstance(y,np.ndarray)
    assert np.all(x.shape == y.shape)
    
    y_min = np.min(y)
    y_max = np.max(y)
    
    
    sampling_rate = 1.0 / np.mean(np.diff(x))
    filtered_signal = savgol_filter(y,10,5)
    
    diff_signal = np.abs(y-filtered_signal)
    
    peaks, properties = scipy.signal.find_peaks(y, distance = 10, height=np.max(y)*min_value_factor, prominence=(0.3, None))
    
    diff_peaks, diff_props = scipy.signal.find_peaks(diff_signal, distance=5, height=np.max(diff_signal)*.3)
    
    plt.plot(y)
    # plt.plot(filtered_signal)
    plt.plot(peaks, y[peaks], "x")
    plt.plot(diff_peaks, y[diff_peaks], "x", 0.3)
    plt.plot(diff_signal)
    
    
    plt.show()
    
    
    


def plot_analysis_results(x:np.ndarray,y:np.ndarray, extrema_dict:dict,
                          out_dir:str, name_extension:str,
                          index:int, index_length:int = 3,
                          x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                          x_lim:list = [0,20], y_lim:list = [0,2500]):
    pass

def save_analysis_results(x:np.ndarray,y:np.ndarray, extrema_dict:dict,
                          out_dir:str, name_extension:str,
                          index:int, index_length:int = 3,
                          x_label: str = "Travel[mm]",y_label: str = "Force[N]"):
    pass



def analyze_parsed_data(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     x_lim:list = [0,20], y_lim:list = [0,2500],
                     save_results = True, save_plot = True):
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
        

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {idx} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {idx} is invalid. (Missing column.)")
            continue
        
        X = np.array(df.get('x').to_list())
        Y = np.array(df.get('y').to_list())

        extrema_results = analyze_data(X,Y)
        
        if save_results:
            save_analysis_results(X,Y,extrema_dict=extrema_results,
                                out_dir=out_dir,
                                name_extension=name_extension,
                                index = idx,
                                index_length=index_length,
                                x_label=x_label,y_label=y_label)
        
        if save_plot:
            plot_analysis_results(X,Y,extrema_dict=extrema_results,
                            out_dir=out_dir,
                            name_extension=name_extension,
                            index = idx,
                            index_length=index_length,
                            x_label=x_label,y_label=y_label,
                            x_lim=x_lim, y_lim=y_lim)
        
        
    
if __name__ == "__main__":
    # Example usage
    svg_path = "sample/20_2500_20_1x1.svg"
    df_list = parse_svg_plot(svg_path)
    
    # save_parsed_data(df_list,"sample/results","result")    
    # save_parsed_data_as_plots(df_list,"sample/results","result")

    
    analyze_parsed_data(df_list,"sample/results","result")