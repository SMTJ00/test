#!/usr/bin/env python
# coding: utf-8

# # Case 2 Maak je eigen app
# 
# Emma Boender, Lars Ossewaarde, Mika Stam, Stijn Schuyt
# 
# Table of contents:
# - Importeer kaggle en benodigde datasets via API
# - Importeer alle benodigde packages
# - Verken data & Feature Engineering
# - Creer app via Streamlit
#     - Opmaak pagina
#     - Vlucht map
#     - Bar chart
#     - Slider met tabel en vlucht map
#     - Eventueel tonen van histogram dmv een checkbox

# ## Importeer kaggle en benodigde datasets via API

# In[1481]:


#import kaggle


# In[1482]:


#api=kaggle.api
#api.get_config_value("username")

#!kaggle datasets download -d ulrikthygepedersen/airlines-delay
#!unzip airlines-delay.zip


# In[1483]:


#!kaggle datasets download -d usdot/flight-delays
#!unzip flight-delays.zip


# ## Importeer alle benodigde packages

# In[1484]:


import pandas as pd
import streamlit as st
import numpy as np
import folium
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import altair as alt
import seaborn as sns


# ## Verken data & Feature Engineering

# Importeer dataset via API

# In[1485]:


file=pd.read_csv('airlines_delay.csv')
# file


# Verken de data:

# In[1486]:


# file.value_counts("Airline")
# file.value_counts("AirportFrom")
# file.value_counts("AirportFrom").head(50)


# Creer een lijst met de 50 grootste airports uit het dataframe, uit de kolom 'AirportFrom'

# In[1487]:


lst1=['ATL', 'ORD', 'DFW', 'DEN','LAX', 'IAH', 'PHX','DTW','LAS', 'SFO', 'CLT','MCO','SLC',
      'MSP', 'EWR', 'JFK', 'BOS', 'BWI', 'LGA', 'SEA','PHL','MDW','DCA','MIA','IAD', 'MEM',
      'SAN','TPA','FLL','STL', 'CLE','CVG','HNL','HOU','BNA','PDX','MCI','RDU','MKE','DAL',
      'OAK','AUS','SNA','SMF','MSY','SJC','SAT','PIT','IND','ABQ']


# Filter dit dataframe op de 50 grootste airports uit de 'AirportFrom' kolom:

# In[1488]:


file_airport=file[file['AirportFrom'].isin(lst1)][['Flight', 'Time', 'Length', 'Airline','AirportFrom', 'AirportTo', 'DayOfWeek', 'Class']]


# Verken de data:

# In[1489]:


# file_airport


# Importeer een een nieuwe dataset met informatie over de airports via API

# In[1490]:


airports=pd.read_csv('airports.csv')


# Verken de dataset

# In[1491]:


# airports.head(30)


# Filter de nieuwe dataset op de 50 gekozen airports en creer hier een nieuw dataframe van:

# In[1492]:


airport_selection=airports[airports['IATA_CODE'].isin(lst1)][['IATA_CODE', 'LATITUDE', 'LONGITUDE']]


# Verken deze dataframe:

# In[1493]:


# airport_selection


# Join deze twee dataframes met elkaar op basis van de IATA code in de kolom 'AirportFrom':

# In[1494]:


table_half_completed = pd.merge(file_airport,airport_selection, left_on='AirportFrom', right_on='IATA_CODE', how='inner')


# Verken de data en sorteer de dataframe op de dag van de week.

# In[1495]:


# table_half_completed.sort_values('DayOfWeek')


# Importeer een nieuwe package voor een dataset met allemaal informatie over airports over de wereld:

# In[1496]:


import airportsdata as ap
airportsdata = ap.load('IATA')


# Maak van deze dataset een dataframe:

# In[1497]:


world_airport = pd.DataFrame(airportsdata)


# Spiegel de dataset zodat de kolommen nu rijen worden en andersom.

# In[1498]:


world_airport = world_airport.transpose()


# Verwijder de index kolom en de kolommen 0,2,3,4,5,6,9 en 10.

# In[1499]:


world_airport.reset_index(drop = True)
world_airport.drop(world_airport.columns[[0,2,3,4,5,6,9,10]],axis=1,inplace=True)


# Join deze nieuwe dataset weer met het eerdere gemaakte dataframe op basis van de IATA code in de kolom 'AirportTo':

# In[1500]:


table_complete = pd.merge(table_half_completed,world_airport, left_on='AirportTo', right_on='iata', how='inner')
table_complete.drop(columns=['iata','IATA_CODE'],inplace=True)


# In[1501]:


# table_complete.sort_values('DayOfWeek')


# In[1502]:


table_complete['DayOfWeekWord']=table_complete['DayOfWeek']


# In[1503]:


# table_complete


# In[1504]:


day_mapping = {1:"Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday", 5:"Friday", 6:"Saturday", 7:"Sunday"}


# In[1505]:


table_complete['DayOfWeekWord']=table_complete['DayOfWeekWord'].map(day_mapping)


# ## Creer app via Streamlit

# ### Opmaak pagina

# In[1506]:


st.title("Airline Database")


# In[1507]:


st.text("!kaggle datasets download -d ulrikthygepedersen/airlines-delay")
st.text("!unzip airlines-delay.zip")

st.text("!kaggle datasets download -d usdot/flight-delays")
st.text("!unzip flight-delays.zip")


# In[1508]:


st.text("Deze pagina geeft een overzicht van bepaalde visualisaties van de gekozen airline op de gekozen dag")


# In[1509]:


# table_complete.value_counts('Airline')


# Creer een lijst met de airlines uit het dataframe, uit de kolom 'Airline'

# In[1510]:


lst2= ['WN', 'DL', 'AA', 'OO', 'US', 'UA', 'MQ', 'XE', 'CO', 'FL', 'EV', 'B6', '9E', 'OH', 'YV', 'AS', 'F9', 'HA']


# Maak een Sidebar met dropdown menu

# In[1511]:


InputAirline=st.sidebar.selectbox('Selecteer airline', (lst2))


# In[1512]:


Airlineselect=table_complete[table_complete["Airline"] == InputAirline]


# In[ ]:





# Maak een selectbox met dropdown menu

# In[1513]:


selectbox=st.selectbox("Selecteer dag van de week", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))


# In[1514]:


Dayselect=Airlineselect[Airlineselect["DayOfWeekWord"]== selectbox]


# In[ ]:





# ### Vlucht map

# In[1515]:


st.text("Hieronder is een map weergegeven waarop alle vluchten te zien zijn van deze airline op deze dag ")


# In[1516]:


df = Dayselect

m = folium.Map(location=[df['LATITUDE'].iloc[0], df['LONGITUDE'].iloc[0]], zoom_start=4)

for _, row in df.iterrows():
    start_point = (row['LATITUDE'], row['LONGITUDE'])
    end_point = (row['lat'], row['lon'])
    line = folium.PolyLine(locations=[start_point, end_point], color='red',weight=1,opacity=0.05)
    line.add_to(m)

for i, row in df.iterrows():
    folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],radius=2,color="black").add_to(m)
    folium.CircleMarker(location=[row['lat'], row['lon']],radius=2,color="black").add_to(m)

st.markdown(folium_static(m), unsafe_allow_html=True)


# In[1517]:


code1 = '''df = Dayselect

m = folium.Map(location=[df['LATITUDE'].iloc[0], df['LONGITUDE'].iloc[0]], zoom_start=4)

for _, row in df.iterrows():
    start_point = (row['LATITUDE'], row['LONGITUDE'])
    end_point = (row['lat'], row['lon'])
    line = folium.PolyLine(locations=[start_point, end_point], color='red',weight=1,opacity=0.05)
    line.add_to(m)

for i, row in df.iterrows():
    folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],radius=2,color="black").add_to(m)
    folium.CircleMarker(location=[row['lat'], row['lon']],radius=2,color="black").add_to(m)

st.markdown(folium_static(m), unsafe_allow_html=True)

'''
st.code(code1, language='python')


# ### Bar chart

# In[1518]:


st.text("Hieronder is een bar chart te zien die per gekozen airline en gekozen dag laat zien wat de verdeling van vluchten die te laat zijn vertrokken en die optijd zijn vertrokken.")


# In[1519]:


df1 = Dayselect.copy()

grouped_counts = df1.groupby(['Class']).size().reset_index()
grouped_counts.columns = ['Class', 'counts']

grouped_counts['Class'] = grouped_counts['Class'].map({0: 'ontime', 1: 'late'})

chart = alt.Chart(grouped_counts).mark_bar().encode(
    x='Class:O',
    y='counts:Q',
    color=alt.Color('Class:N', scale=alt.Scale(range=['red', 'green']))
)

chart = chart.properties(
    width=alt.Step(80),
    height=alt.Step(200)
)

st.altair_chart(chart, use_container_width=True)


# ### Slider met tabel en vlucht map

# In[1520]:


st.text("Via onderstaande sliders kan er een bepaalde range van vluchtlengte gekozen worden. Er wordt dan een overzicht gecreeërd met alle vluchten van deze airline op deze dag binnen deze range.")


# In[ ]:





# In[1521]:


def multiply60(num):
    decimal_part = num - int(num)   # extract the decimal part of the number
    new_decimal_part = decimal_part * 60   # multiply the decimal part by 60
    new_num = int(num) + new_decimal_part/100   # add the new decimal part back to the integer part
    return round(new_num,2)


# In[1522]:


df123 = Dayselect.copy()


df123['Length'] = round((df123['Length']/60),2)

min_len = df123['Length'].min()
max_len = df123['Length'].max()

selected_min_len, selected_max_len = st.slider(
    'Select a range of flight length in hour:',
    min_value=min_len, max_value=max_len, value=(min_len, max_len))

df123['Time'] = round((df123['Time']/60),2)
df123['Time'] = df123['Time'].apply(multiply60)


min_len_time = df123['Time'].min()
max_len_time = df123['Time'].max()

selected_min_len_time, selected_max_len_time = st.slider(
    'Select a range of depatures:',
    min_value=min_len_time, max_value=max_len_time, value=(min_len_time, max_len_time))

st.write(f"Flights with length between {round(selected_min_len,2)}h and {round(selected_max_len,2)}h:")
filtered_df = df123[(df123['Length'] >= selected_min_len) & (df123['Length'] <= selected_max_len)]

st.write(f"Flights with length between {(selected_min_len_time)} and {(selected_max_len_time)}:")
filtered_df1 = filtered_df[(filtered_df['Time'] >= selected_min_len_time) & (filtered_df['Time'] <= selected_max_len_time)]
st.write(filtered_df1)


# In[1523]:


st.text("Op basis van de bovenstaande gekozen ranges wordt ook de map geupdate met deze vluchten.")


# In[1524]:


try:
    df = filtered_df1

    m = folium.Map(location=[df['LATITUDE'].iloc[0], df['LONGITUDE'].iloc[0]], zoom_start=4)

    for _, row in df.iterrows():
        start_point = (row['LATITUDE'], row['LONGITUDE'])
        end_point = (row['lat'], row['lon'])
        line = folium.PolyLine(locations=[start_point, end_point], color='red',weight=1,opacity=0.05)
        line.add_to(m)

    for i, row in df.iterrows():
        folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],radius=2,color="black").add_to(m)
        folium.CircleMarker(location=[row['lat'], row['lon']],radius=2,color="black").add_to(m)

    st.markdown(folium_static(m), unsafe_allow_html=True)
except IndexError:
    print("No data")


# ### Eventueel tonen van een histogram dmv een checkbox

# In[1525]:


st.text("Je kan hieronder als eventuele optie een histogram laten zien. Klik op Toon histogram.")
st.text("Dit histogram laat zien voor de gekozen airline welke vluchten hoelaat vertrekken met hun bijbehorende vluchtlengte.")
st.text("Hoe donker de kleur, hoe groter de frequentie van het aantal vluchten op dat tijdstip met die vluchtlengte.")


# In[1526]:


df_test = Dayselect.copy()
df_test['Time'] = round((df_test['Time']/60),2)
print(df_test)
show_histogram=False

show_histogram=st.checkbox("Toon histogram")

if show_histogram:
    x_axis="Time"
    y_axis="Length"
    
    sns.set_style('darkgrid')
    fig,ax=plt.subplots()
    sns.histplot(df_test, x=x_axis, y=y_axis, kde=False, ax=ax)
    ax.set_xlabel("Vertrek tijd (h)")
    ax.set_ylabel("Vluchtlengte (min)")
    ax.set_title("Hoeveelheid vluchtenvertrek per vluchtlengte")
    st.pyplot(fig)
    code2 = '''if show_histogram:
    x_axis="Time"
    y_axis="Length"
    
    sns.set_style('darkgrid')
    fig,ax=plt.subplots()
    sns.histplot(Dayselect, x=x_axis, y=y_axis, kde=False, ax=ax)
    st.pyplot(fig)'''
    st.code(code2, language='python')
    
    st.text('Heeft deze pagina voldoende informatie gegeven?')
    label1='Ja!'
    label2='Nee.'
    Ja=st.checkbox(label1, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, 
                      disabled=False, label_visibility="visible")
    Nee=st.checkbox(label2, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, 
                      disabled=False, label_visibility="visible")
    if Ja:
        st.write('Thanks!')
    if Nee:
        st.write('Oke, bedankt.')


# In[1527]:


def create_histogram(column, show_trendline):
    plt.clf()
    histo = Dayselect.copy()
    histo['Time'] = round((histo['Time']/60),2)
    histo['Time'] = histo['Time'].apply(multiply60)
    sns.histplot(histo["Time"], kde=False)

    if show_trendline:
        counts, bins = np.histogram(histo["Time"], bins= "auto")
        line = plt.plot(bins[:-1], counts, linestyle='-',color='red')

    plt.title('Verdeling vlucht vertrekken over de dag')
    plt.xlabel('Time (h)')
    st.pyplot()
      
def main():
    st.title('Flight Counts')
    show_trendline = st.checkbox('Show line')
    create_histogram("Flight", show_trendline)
    
main()
    
st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:




