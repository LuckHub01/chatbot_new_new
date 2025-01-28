import streamlit as st
import sqlite3
import time
import math
import pandas as pd
from geopy.distance import geodesic
from urllib.request import urlopen
import json


# Fonction pour lire les données de la base de données SQLite
def read_gendarmeries_database():
    conn = sqlite3.connect('gendarmeries.db')
    query = "SELECT name, latitude, longitude FROM gendarmeries"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Fonction pour calculer la distance entre deux points
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Fonction pour générer le lien Google Maps
def generate_google_maps_link(lat, lon, destination_lat, destination_lon):
    #google_maps_link = f"https://www.google.com/maps?q={destination_lat},{destination_lon}"
    google_maps_link = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={destination_lat},{destination_lon}&travelmode=driving"
    return google_maps_link
    #return f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={destination_lat},{destination_lon}"


# Fonction pour générer l'URL de l'image OpenStreetMap
def generate_openstreetmap_image_url(lat, lon, zoom=15, width=600, height=300):
    return f"https://maps.locationiq.com/v3/staticmap?key=YOUR_LOCATIONIQ_API_KEY&center={lat},{lon}&zoom={zoom}&size={width}x{height}&format=png&maptype=roadmap&markers=size:mid%7Ccolor:red%7Clabel:G%7C{lat},{lon}"

def get_user_location():
    url="http://ipinfo.io/json"
    response=urlopen(url)
    data=json.load(response)
    coord_str=data['loc']
    coordinates = [float(x) for x in coord_str.split(",")]
    #liste=data['loc'].split(",")
    #liste=data['loc'].list()
    user_lat=coordinates[0]
    user_lon=coordinates[1]
    return user_lat, user_lon

    
# Formule de Haversine pour calculer la distance entre deux points (en km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance en km

user_lat, user_lon = get_user_location()
# # Fonction pour trouver la gendarmerie la plus proche
# 

print(get_user_location())

def find_nearest_station(user_lat, user_lon, gendarmeries):
    gendarmeries['distance'] = gendarmeries.apply(
        lambda row: calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
    )
    nearest_gendarmerie = gendarmeries.loc[gendarmeries['distance'].idxmin()]
    return (nearest_gendarmerie['name'], nearest_gendarmerie['latitude'], nearest_gendarmerie['longitude']), nearest_gendarmerie['distance']


#print(find_nearest_station(user_lat, user_lon))

