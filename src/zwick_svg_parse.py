import os, sys, glob, re
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import scipy.integrate
import scipy.signal

import scipy.stats
from sklearn.metrics import mean_squared_error

from svgelements import SVG, Path, Text, Line

import scipy
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter


def parse_svg_plot(svg_path:str, min_segment_len:int = 100, 
                   x_min:float = 0, y_min:float = 0, 
                   x_max:float = 20, y_max:float =2500,
                   x_invert:bool = False, y_invert:bool = False,
                   start_skip = 5, end_skip = 5,
                   x_start_limit = 1,
                   *args, **kwargs)-> List[pd.DataFrame]:
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
            
        if x[0]>x_start_limit:
            continue

        df = pd.DataFrame({"x":list(x),"y":list(y)})
        
        df_list.append(df)

    return df_list


def save_parsed_data(parsed_data:List[pd.DataFrame],
                     out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     index_mapping_dict: Optional[dict] = None, *args, **kwargs):
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
        
    col_lookup = {"x":x_label,"y":y_label}

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if isinstance(index_mapping_dict,dict): # maps indices to an ID
            sample_name = index_mapping_dict.get(idx)
        else:
            sample_name = str(idx).zfill(index_length)
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {idx} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {idx} is invalid. (Missing column.)")
            continue
        
        _df = df.copy()
        _df.columns = [col_lookup.get(c) for c in df.columns]
        
        out_path = os.path.join(out_dir,f"{name_extension}_{sample_name}.csv")
        _df.to_csv(out_path,index = False, float_format="%.3f")
        
        print(f"Sample {sample_name} saved at {out_path}")
        

def save_parsed_data_as_plots(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     x_lim:list = [0,20], y_lim:list = [0,2500],
                     index_mapping_dict: Optional[dict] = None,*args, **kwargs):
    
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
        
    col_lookup = {"x":x_label,"y":y_label}

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if isinstance(index_mapping_dict,dict): # maps indices to an ID
            sample_name = index_mapping_dict.get(idx)
        else:
            sample_name = str(idx).zfill(index_length)
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {sample_name} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {sample_name} is invalid. (Missing column.)")
            continue
        
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'])
        ax.set_xlabel(col_lookup.get('x'))
        ax.set_ylabel(col_lookup.get('y'))
        
        ax.set_title(f'Sample {sample_name}')
        
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
        
        out_path = os.path.join(out_dir,f"{name_extension}_{sample_name}.png")
        
        plt.savefig(out_path)  # Save the figure
        plt.close()
        
        print(f"Plot created for sample {sample_name} at {out_path}")
        
        
def analyze_data(x:np.ndarray,y:np.ndarray, target_num_of_extremas:int=2, min_value_factor:float = 0.2,*args, **kwargs)->dict:
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
    

# Local extrema analysis
def local_analyze_parsed_data(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     x_lim:list = [0,20], y_lim:list = [0,2500],
                     save_results = False, save_plot = False,*args, **kwargs):
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
            plot_local_analysis_results(X,Y,extrema_dict=extrema_results,
                                out_dir=out_dir,
                                name_extension=name_extension,
                                index = idx,
                                index_length=index_length,
                                x_label=x_label,y_label=y_label)
        
        if save_plot:
            save_local_analysis_results(X,Y,extrema_dict=extrema_results,
                            out_dir=out_dir,
                            name_extension=name_extension,
                            index = idx,
                            index_length=index_length,
                            x_label=x_label,y_label=y_label,
                            x_lim=x_lim, y_lim=y_lim)
            

def plot_local_analysis_results(x:np.ndarray,y:np.ndarray, extrema_dict:dict,
                          out_dir:str, name_extension:str,
                          index:int, index_length:int = 3,
                          x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                          x_lim:list = [0,20], y_lim:list = [0,2500]):
    raise NotImplementedError()

def save_local_analysis_results(x:np.ndarray,y:np.ndarray, extrema_dict:dict,
                          out_dir:str, name_extension:str,
                          index:int, index_length:int = 3,
                          x_label: str = "Travel[mm]",y_label: str = "Force[N]"):
    raise NotImplementedError()

        
        
def compare_maximum_values(df_list,max_val_list,*args, **kwargs):
    x = [] # orig
    y= [] # approx
    
    for df,max_val in zip(df_list,max_val_list):
        y.append(np.max(df["y"]))
        x.append(max_val)
    
    x = np.array(x)
    y = np.array(y)

    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    y_pred = slope * x + intercept

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Calculate SE of residuals
    residuals = y - y_pred
    se = scipy.stats.sem(residuals)

    # Plot
    plt.scatter(x, y)
    plt.plot(x, y_pred, color='red')

    # Annotate metrics
    plt.annotate(
        f"y = {slope:.2f}x + {intercept:.2f}\n"
        f"$R^2$ = {r_value**2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"SE = {se:.3f}",
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=12, ha='left', va='top'
    )

    plt.xlabel("Orig Fmax (N)")
    plt.ylabel("Approx. Fmax (N)")
    plt.title("ZWICK svg parse precision")
    plt.show()



def sigmoid(x, x0, k, L, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

def tangent_line(x, x0, k, L, b):
    slope = L * k / 4
    y_mid = L/2 + b  # Sigmoid value at x=x0
    return slope * (x - x0) + y_mid

def curve_characteristics(parsed_data:List[pd.DataFrame],out_dir:str, name_extension:str, start_index:int = 1, index_length:int = 3,
                     x_label: str = "Travel[mm]",y_label: str = "Force[N]",
                     x_lim:list = [0,20], y_lim:list = [0,2500],
                     save_plot = True, show_plot = False,
                     index_mapping_dict: Optional[dict] = None, 
                     analysis_x_range: Optional[list] = None,
                     *args, **kwargs):
    
    assert isinstance(parsed_data,list)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=False)
    
    res_list = []
    

    for i, df in enumerate(parsed_data):
        idx = start_index+i
        
        if isinstance(index_mapping_dict,dict): # maps indices to an ID
            sample_name = index_mapping_dict.get(idx)
        else:
            sample_name = str(idx).zfill(index_length)
        
        if not isinstance(df,pd.DataFrame):
            print(f"Sample {idx} is invalid. (Invalid type.)")
            continue
        
        if not all([c in df.columns for c in ["x","y"]]):
            print(f"Sample {idx} is invalid. (Missing column.)")
            continue
        
        X = np.array(df.get('x').to_list())
        Y = np.array(df.get('y').to_list())
        
        # sanitize: remove data points when X step is 0
        sanitize_mask = np.diff(X) > 0
        x_lim_mask = X>0
        X = X[:-1][sanitize_mask&x_lim_mask[:-1]]
        Y = Y[:-1][sanitize_mask&x_lim_mask[:-1]]
        
        if isinstance(analysis_x_range,list):
            assert len(analysis_x_range)==2
            analysis_x_range = sorted(analysis_x_range)
            indices = np.where((X >= analysis_x_range[0]) & (X <= analysis_x_range[1]))[0]
            assert len(indices)>0, f"There is no data on the range of {analysis_x_range}"
            
            X = X[indices]
            Y = Y[indices]

            x_lim = [max(x_lim[0],analysis_x_range[0]),min(x_lim[1],analysis_x_range[1])]
            
        
        # global maxima
        F_max = np.max(Y)
        F_max_index = np.argmax(Y)
        F_max_loc = X[F_max_index]
        
        curve_x_before = X[:F_max_index+1]        
        curve_y_before = Y[:F_max_index+1]

        curve_x_after = X[F_max_index:]        
        curve_y_after = Y[F_max_index:]
        
        
        # stored energy
        area_total_simpson = scipy.integrate.simpson(Y,X) 
        area_before_simpson =  scipy.integrate.simpson(curve_y_before,curve_x_before)
        area_after_simpson =  scipy.integrate.simpson(curve_y_after,curve_x_after)
        
        area_total_trapz = np.trapezoid(Y,X)
        area_before_trapz =  np.trapezoid(curve_y_before,curve_x_before)
        area_after_trapz =  np.trapezoid(curve_y_after,curve_x_after)
        
        # gradient
        before_grad = np.gradient(curve_y_before,curve_x_before)
        before_grad_max = np.max(before_grad)
        before_gra_max_loc = curve_x_before[np.argmax(before_grad)]
        
        try:
            after_grad = np.gradient(curve_y_after,curve_x_after)
            after_grad_min = np.min(after_grad)
            after_grad_min_loc = curve_x_after[np.argmin(after_grad)]
        except Exception as exc:
            print(exc)
            after_grad = np.nan
            after_grad_min = np.nan
            after_grad_min_loc = np.nan
        
        # sigmoid fit
        
        # initial params
        
        L_estimate = np.max(curve_y_before) - np.min(curve_y_before)
        b_estimate = np.min(curve_y_before)
        k_estimate = 5
        x0_estimate = np.median(curve_x_before)  # or where gradient is maximum
        p0 = [x0_estimate, k_estimate, L_estimate, b_estimate]
                
        popt, pcov = scipy.optimize.curve_fit(sigmoid, curve_x_before, curve_y_before, p0 = p0, maxfev=10000)
        f_approx = sigmoid(curve_x_before, *popt)
        f_approx_rmse = np.sqrt(mean_squared_error(curve_y_before, f_approx))
        x0, k, L, b = popt
                
        # fit linear around inflexion point       
        linear_slope = L * k / 4 # Slope of linear region        
        y_mid = L/2 + b # Midpoint y-value
        eqn = f"y = {linear_slope:.2f}(x - {x0:.2f}) + {y_mid:.2f}" # Linear equation string
        
        linear_mask = (curve_x_before >= x0 - 2/k) & (curve_x_before <= x0 + 2/k)
        linear_x = curve_x_before[linear_mask]
        linear_y = curve_y_before[linear_mask]
        
        # Plot validation
        plt.figure(figsize=(12, 8))
        
        # Main curves
        plt.plot(X, Y, label='Original Curve', color='blue', linewidth=1, alpha = 0.5)
        plt.plot(curve_x_before, f_approx, label='Fitted Sigmoid', color='green', linewidth=1)
        plt.plot(linear_x, tangent_line(linear_x, x0, k, L, b), 
                'r--', label='Tangent from Sigmoid Fit', linewidth=1)
        
        plt.fill_between(curve_x_before, 0, curve_y_before, color='orange', alpha=0.3, label='Before Max Energy [Nm]')
        plt.fill_between(curve_x_after, 0, curve_y_after, color='purple', alpha=0.3, label='After Max Energy [Nm]')
        
        # Annotate sigmoid midpoint
        y_at_midpoint = sigmoid(x0, x0, k, L, b)
        plt.annotate(f'Midpoint (Sigmoid x0={x0:.2f})', 
                    xy=(x0, y_at_midpoint),
                    xytext=(x0 + 0.1, y_at_midpoint - 100),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Annotate curve max
        plt.annotate(f'Max: {F_max:.1f}N @ {F_max_loc:.2f}mm', 
                    xy=(F_max_loc, F_max),
                    xytext=(F_max_loc - 0.15, F_max + 100),
                    arrowprops=dict(facecolor='purple', arrowstyle='->'),
                    fontsize=10, color='purple',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Vertical dotted line at curve max location
        plt.axvline(F_max_loc, color='purple', linestyle=':', 
                   label='Curve Max Location', linewidth=1)

        
        
        # Calculate energy ratios
        ratio_before = area_before_simpson / area_total_simpson if area_total_simpson != 0 else 0
        ratio_after = area_after_simpson / area_total_simpson if area_total_simpson != 0 else 0
        
        # Energy annotation box
        energy_text = f'Energy Analysis:\nTotal: {area_total_simpson/1000.0:.1f} Nm\nBefore Max: {area_before_simpson/1000.0:.1f} Nm ({ratio_before:.1%})\nAfter Max: {area_after_simpson/1000.0:.1f} Nm ({ratio_after:.1%})'
        plt.text(0.02, 0.95, energy_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Fit and parameters:
        textstr = f'Sigmoid RMSE: {f_approx_rmse:.2f}\nSigmoid k: {k:.2f}\nTangent Slope: {linear_slope:.2f} N/mm'
        plt.text(0.02, 0.80, textstr, transform=plt.gca().transAxes, fontsize=9,
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Labels, legend, grid, limits
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.title(f'Curve Characteristics - Sample {sample_name}', fontsize=14)
        
        
        if save_plot:
            plt.savefig(os.path.join(out_dir, f'sample_{sample_name}.png'), 
                       dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()

        res_list.append({"sample":sample_name,
                         "F_max":F_max,
                         "F_max_loc":F_max_loc,
                         "Total_energy":area_total_simpson,
                         "Before_F_max_energy":area_before_simpson,
                         "After_F_max_energy":area_after_simpson,
                         "Before_Grad_max":before_grad_max,
                         "Before_Grad_max_loc":before_gra_max_loc,
                         "After_grad_min":after_grad_min,
                         "After_grad_min_loc":after_grad_min_loc,
                         "sigmoid_x0":x0,
                         "sigmoid_k":k,
                         "sigmoid_L":L,
                         "sigmoid_b":b,
                         "sigmoid_RMSE":f_approx_rmse,
                         "sigmoid_linear_slope":linear_slope
                         })
        
    res_df = pd.DataFrame(res_list)
    res_df.to_csv(os.path.join(out_dir,f"{name_extension}.csv"),index=False,float_format="%.3f")


def zwick_parse_pipeline(svg_path:str, settings:dict, out_dir:str, save_plots:bool =True):
    assert os.path.exists(svg_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    df_list = parse_svg_plot(svg_path,**settings)
    save_parsed_data(df_list,os.path.join(out_dir,"curves"),"sample",**settings)    
    if save_plots:
        save_parsed_data_as_plots(df_list,os.path.join(out_dir,"curves"),"sample",**settings)
    
    curve_characteristics(df_list,os.path.join(out_dir,"curve_characteristics"),"result",save_plot=save_plots, **settings)
    

if __name__ == "__main__":
    settings = OrderedDict(x_label= "Travel[mm]",y_label = "Force[N]",
                x_lim= [0,20], y_lim= [0,3000],x_min= 0, y_min = 0, 
                x_max= 20, y_max=3000, min_segment_len = 200)
    
    
    # Example usage
    # svg_path = "sample/20_2500_20_1x1.svg"
    # df_list = parse_svg_plot(svg_path,**settings)
    
    # save_parsed_data(df_list,"sample/curves","sample",**settings)    
    # save_parsed_data_as_plots(df_list,"sample/curves","sample",**settings)
       
    # F max precision measure
    # ref_df = pd.read_csv('sample/references.csv',sep=";")
    # ref_df.columns = [str(c).strip() for c in ref_df.columns]
    # max_val_list = ref_df["Fmax."].apply(lambda x: float(str(x).split(" ")[0].replace(",","."))).to_list()
    # compare_maximum_values(df_list,max_val_list)    
    
    # Local extrema analysis
    # local_analyze_parsed_data(df_list,"sample/local_analysis","result")
    
    # Curve characteristics
    # curve_char_df = curve_characteristics(df_list,"sample/curve_characteristics","result",**settings)
    
    
    # zwick_parse_pipeline(svg_path,settings,"sample")
    
    # zwick_parse_pipeline("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/mc_bones_all.svg",
    #                     settings,
    #                     "/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/all_res")
    
    met_index_df = pd.read_csv("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/met_order.csv")
    met_index_dict = dict(zip(met_index_df['index'], met_index_df['sample_id']))
    
    met_settings = OrderedDict(x_label= "Travel[mm]",y_label = "Force[N]",
            x_lim= [0,20], y_lim= [0,650],x_min= 0, y_min = 0, 
            x_max= 20, y_max=650, min_segment_len = 200,
            index_mapping_dict = met_index_dict,
            analysis_x_range = [0,12.5])
    
    zwick_parse_pipeline("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/raw/MET_20x650.svg",
                        met_settings,
                        "/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/MET",
                        )
    
    rib_index_df = pd.read_csv("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/rib_oder.csv")
    rib_index_dict = dict(zip(rib_index_df['index'], rib_index_df['sample_id']))
        
    rib_settings = OrderedDict(x_label= "Travel[mm]",y_label = "Force[N]",
        x_lim= [0,15], y_lim= [0,300],x_min= 0, y_min = 0, 
        x_max= 15, y_max=300, min_segment_len = 200,
        index_mapping_dict = rib_index_dict,
        analysis_x_range = [0,12.5])
        
    zwick_parse_pipeline("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/raw/RIB_15x300.svg",
                        rib_settings,
                        "/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/piglet/bone_fracture/RIB")