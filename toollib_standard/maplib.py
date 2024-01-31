# Add incidents into map
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')


from concurrent.futures import ThreadPoolExecutor
from geopy.distance import great_circle
import folium


def station_map_add(stalist,lats,lons,this_map,c,cr):
    # this function adds incidents into the map based on an existing map plot
    # stalist: the dictionary record station name
    # c:color tag 01
    # cr: color tag 02
    incidents = folium.map.FeatureGroup()
    latitudes = list(lats)
    longitudes = list(lons)
    labels = list(stalist)
    for lat, lng, label in zip(latitudes, longitudes, labels):
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                popup = label,
                radius=2.5, # define how big you want the circle markers to be
                color=c,
                fill=True,
                fill_color=cr,
                fill_opacity=0.8
            )
        )
    # Add incidents to map
    this_map.add_child(incidents)
    return this_map

def lat_lon(sta_info,c,cr,this_map):
    
    # this function plot the map with incidents according to sta_info 
    # based on an existing map plot
    # sta_info: the dictionary that record station name, lat, and lon.
    lats = []
    lons = []
    #days = []
    for sta_name in list(sta_info.keys()):
        lats.append(sta_info[sta_name]['lat'])
        lons.append(sta_info[sta_name]['lon'])
        #days.append([sta_info[sta_name]['d_start'],sta_info[sta_name]['d_end']])
    this_map = station_map_add(sta_info.keys(),lats,lons,this_map,c,cr)
    return this_map

def plot_single_area(sta_list,key_loc,cs,crs,zoom_rate):
    # this function plot the map with incidents with a given location center
    # which generate a new map plot
    # 
    # key_loc: the standard flag to find the center of the map plot
    loc_standard = list(sta_list.keys())[key_loc]
    yy = sta_list[loc_standard]['lon']
    xx = sta_list[loc_standard]['lat']
    world_map = folium.Map()
    this_map = folium.Map(location=[xx,yy], zoom_start=zoom_rate)
    this_map = lat_lon(sta_list,cs[0],crs[0],this_map)
    return this_map

# color bars
cs = ['black','blue','green','purple','red','orange']
crs = ['white','red','green','yellow']