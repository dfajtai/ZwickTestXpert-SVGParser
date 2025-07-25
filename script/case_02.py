from src.zwick_svg_parse import *

if __name__ == "__main__":    
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