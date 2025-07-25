from src.zwick_svg_parse import *

if __name__ == "__main__":
    settings = OrderedDict(x_label= "Travel[mm]",y_label = "Force[N]",
            x_lim= [0,20], y_lim= [0,3000],x_min= 0, y_min = 0, 
            x_max= 20, y_max=3000, min_segment_len = 200)
    
    zwick_parse_pipeline("/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/mc_bones_all.svg",
                        settings,
                        "/nas/medicopus_share/Projects/ANIMALS/PIGWEB_TNA/mc_bones/measurements/all_res")