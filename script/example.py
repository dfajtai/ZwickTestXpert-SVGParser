from src.zwick_svg_parse import *


if __name__ == "__main__":
    settings = OrderedDict(x_label= "Travel[mm]",y_label = "Force[N]",
            x_lim= [0,20], y_lim= [0,3000],x_min= 0, y_min = 0, 
            x_max= 20, y_max=3000, min_segment_len = 200)
    
    # Example usage
    svg_path = "sample/20_2500_20_1x1.svg"
    df_list = parse_svg_plot(svg_path,**settings)
    
    save_parsed_data(df_list,"sample/curves","sample",**settings)    
    save_parsed_data_as_plots(df_list,"sample/curves","sample",**settings)
       
    # F max precision measure
    ref_df = pd.read_csv('sample/references.csv',sep=";")
    ref_df.columns = [str(c).strip() for c in ref_df.columns]
    max_val_list = ref_df["Fmax."].apply(lambda x: float(str(x).split(" ")[0].replace(",","."))).to_list()
    compare_maximum_values(df_list,max_val_list)    
    
    # Local extrema analysis
    local_analyze_parsed_data(df_list,"sample/local_analysis","result")
    
    # Curve characteristics
    curve_char_df = curve_characteristics(df_list,"sample/curve_characteristics","result",**settings)
    
    zwick_parse_pipeline(svg_path,settings,"sample")


    # zwick_parse_pipeline("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/mc_bones_all.svg",
    #                     settings,
    #                     "/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/all_res")