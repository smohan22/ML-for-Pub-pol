import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup
import urllib
import urllib3
import ast
# downloaded data using :https://data.cityofchicago.org/browse?browseSearch=311&view_type=rich&q=311&sortBy=relevance&utf8=%E2%9C%93
def acquisition():
    #Get raw data from the 311 Chicago request portal csv files
    graffitti = pd.read_csv("Graffiti_Removal.csv")
    pot_holes = pd.read_csv("Pot_Holes_Reported.csv")
    sanitation = pd.read_csv("Sanitation_Code_Complaints.csv")
    building = pd.read_csv("Vacant_and_Abandoned_Buildings_Reported.csv")
    orig_col = ["Creation Date", "Status", "Completion Date", "Service Request Number", "Type of Service Request", "Street Address", "ZIP Code", "X Coordinate", "Y Coordinate", "Ward", "Police District", "Community Area", "Latitude", "Longitude", "Location"]

    #Clean up the data to create a more systematic and useful file
    graffitti = graffitti[["Creation Date", "Completion Date", "Service Request Number", "Type of Service Request", "What Type of Surface is the Graffiti on?", "Street Address", "ZIP Code", "X Coordinate", "Y Coordinate", "Ward", "Police District", "Community Area", "Latitude", "Longitude", "Location"]]
    pot_holes = pot_holes[["CREATION DATE", "COMPLETION DATE", "SERVICE REQUEST NUMBER", "TYPE OF SERVICE REQUEST", "STREET ADDRESS", "ZIP", "X COORDINATE", "Y COORDINATE", "Ward", "Police District", "Community Area", "LATITUDE", "LONGITUDE", "LOCATION"]]
    pot_holes.columns = ["Creation Date", "Completion Date", "Service Request Number", "Type of Service Request", "Street Address", "ZIP Code", "X Coordinate", "Y Coordinate", "Ward", "Police District", "Community Area", "Latitude", "Longitude", "Location"]
    sanitation = sanitation[["Creation Date", "Service Request Number", "Type of Service Request", "What is the Nature of this Code Violation?", "Street Address", "ZIP Code", "X Coordinate", "Y Coordinate", "Ward", "Police District", "Community Area", "Latitude", "Longitude", "Location"]]
    building.columns = [col_name.title() for col_name in building.columns]
    building.rename(columns={"Zip Code": "ZIP Code", 'Service Request Type': 'Type of Service Request', 'Date Service Request Was Received': 'Creation Date'}, inplace=True)
    building["Street Address"] = building["Address Street Number"].map(str) + " " + building["Address Street Direction"] + " " + building["Address Street Name"] + " " + building["Address Street Suffix"]
    building = building[["Creation Date", "Service Request Number", "Type of Service Request", "Street Address", "ZIP Code", "X Coordinate", "Y Coordinate", "Ward", "Police District", "Community Area", "Latitude", "Longitude", "Location"]]

    #create combined files with data from all 4 fiiles
    combined_311 = graffitti.append([pot_holes, sanitation, building])
    combined_311["Response Time"] = pd.to_numeric(pd.to_datetime(combined_311["Completion Date"])- pd.to_datetime(combined_311["Creation Date"]))
    combined_311["complaint_month"] = pd.DatetimeIndex(combined_311["Creation Date"]).month 

    combined_311.to_csv("Result.csv")

    #Create tables from the data available

    count_311_combined = combined_311["Type of Service Request"].value_counts()
    count_311_combined[:].plot("bar")
    count_311_pivot = pd.pivot_table(combined_311, index = ["complaint_month"], columns = ["Type of Service Request"],values = ["Creation Date"], aggfunc = [len])
    plt.plot(count_311_pivot[[0]], color = 'red')
    plt.plot(count_311_pivot[[1]], color = 'blue')
    plt.plot(count_311_pivot[[2]], color = 'green')
    plt.plot(count_311_pivot[[3]], color = 'yellow')
    plt.xticks(range(12), ["Jan", "Feb", 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], size = 'small')
    plt.legend(prop={'size':5})
    plt.savefig("combined_monthwise.jpeg")

    count_311_response = pd.pivot_table(combined_311, index = ["Response Time"], columns = ["Type of Service Request"],values = ["Creation Date"], aggfunc = [len])

    count_ward = combined_311.groupby(["Ward"],axis = 0).count()["Creation Date"]
    count_ward.plot("bar")
    plt.savefig("combined_wardwise.jpeg")

    #Ward wise data
    count_311_combined = pd.pivot_table(combined_311, index = ["Ward"], columns = ["Type of Service Request"],values = ["Creation Date"], aggfunc = [len])
    t = count_311_combined
    plt.bar(np.arange(0,51), [x[0] for x in t[[0]].values.tolist()], width = 0.5, color = 'red', label = "Graffiti Removal" )
    plt.bar(np.arange(0,51)+ 1, [x[0] for x in t[[1]].values.tolist()], width = 0.5, color = 'blue', label = "Pothole in Street")
    plt.bar(np.arange(0,51)+2, [x[0] for x in t[[2]].values.tolist()],  width = 0.5, color = 'green', label = "Sanitation Code Violation")
    plt.bar(np.arange(0,51)+3, [x[0] for x in t[[3]].values.tolist()],  width = 0.5,  color = 'yellow', label = "Vacant/Abandoned Building")
    plt.xticks(range(51))
    plt.legend(prop={'size':5})
    plt.tight_layout()
    plt.savefig("combined_requestwise.jpeg")


    #Details for Grafitti:
    count_graffitti = graffitti["What Type of Surface is the Graffiti on?"].value_counts()
    count_graffitti.plot("bar")
    graffitti['Latitude'].plot('hist', bins=50)
    plt.savefig("type_of_grafitti.jpeg")


    count_graffitti_ward = combined_311[combined_311["Type of Service Request"] == "Graffiti Removal"].groupby(["Ward"],axis = 0).count()["Creation Date"]
    count_graffitti_ward.plot("bar")
    plt.savefig("grafitti_type_wardwise.jpeg")


    #Details for Sanitation
    count_sanitation = sanitation["What is the Nature of this Code Violation?"].value_counts()
    count_sanitation.plot("bar")
    plt.savefig("sanitataion_issuewise.jpeg")

    
    count_sanitation_ward = combined_311[combined_311["Type of Service Request"] == "Sanitation Code Violation"].groupby(["Ward"],axis = 0).count()["Creation Date"]
    count_sanitation_ward.plot("bar")
    plt.savefig("sanitation_wardwise.jpeg")






##########################################################################Part 2 ######################################################################

apikey = "1b405de29a6fab983a2375f6aa4af12ff59082e5"

code_represents = {"B01002_001E": "age",  #median age
                    "B23025_002E": "employment_labor_force", #Number of persons, age 16 or older, in the labor force.,
                    "B23025_005E": "employment_unemployed", #Number of unemployed, age 16 or older, in the civilian labor force. 
                    "B19013_001E": "income", # Median household income in the past 12 months (in 2013 inflation-adjusted dollars).
                    "B19301_001E": "income_per_capita", #Per capita income in the past 12 months (in 2013 inflation-adjusted dollars).
                    "B01002_003E": "median_female_age", #Median age by sex (female).
                    "B01002_002E": "median_male_age", #Median age by sex (male).
                    "B01003_001E": "population", #Total population
                    "B17001_002E":  "poverty", #Number of persons whose income in the past 12 months is below the poverty level
                    }

def main():

    combined_311 = pd.read_csv("Result.csv")
    #For Sanitation
    req_data = combined_311[combined_311["Type of Service Request"] == "Sanitation Code Violation"]
    req_data_san = req_data.pivot_table(index=['Location'], aggfunc='count')
    req_data_san_top = req_data_san.sort_values(by=['Creation Date'], ascending=False)[:10]
    req_data_san_bottom = req_data_san.sort_values(by=['Creation Date'], ascending=False)[-10:]
    most_complaints_san = pd.DataFrame(augment(req_data_san_top))
    least_compaints_san = pd.DataFrame(augment(req_data_san_bottom))
    
    #For Vacant Building
    req_data = combined_311[combined_311["Type of Service Request"] == "Vacant/Abandoned Building"]
    req_data_bldg = req_data.pivot_table(index=['Location'], aggfunc='count')
    req_data_bldg_top = req_data_bldg.sort_values(by=['Creation Date'], ascending=False)[:10]
    req_data_bldg_bottom = req_data_san.sort_values(by=['Creation Date'], ascending=False)[-10:]
    most_complaints_bldg = pd.DataFrame(augment(req_data_bldg_top))
    least_compaints_bldg = pd.DataFrame(augment(req_data_bldg_bottom))
    
    #return most_complaints_san, least_compaints_san, most_complaints_bldg , least_compaints_bldg 
    cook_county_data = cook_county()

    #sample graph shown for age
    #we compare the max and min with cook county data taken as average
    #Graph for Sanitation
    l1 =plt.bar(np.arange(0,10), [float(x) for x in most_complaints_san['age'].values.tolist()], width = .4, color = 'red', label = "max_compaints" )
    l2 = plt.bar(np.arange(0,10)+ 0.5, [float(x) for x in least_compaints_san['age'].values.tolist()], width = .4 , color = 'blue', label = "min_complaints")
    plt.axhline(y = float(cook_county_data["age"][0]))
    plt.xticks(range(11))
    plt.legend(prop={'size':5})
    plt.tight_layout()
    plt.xlabel("for max complaints 10 = min, for min complaints 10 = min")
    plt.ylabel("Average Age")
    plt.title('Sanitation Complaints per Demographics (age)')
    plt.savefig("sanitation_demographics.jpeg")

    #Graph for Vacant/Abandoned Buildings:
    l1 =plt.bar(np.arange(0,10), [float(x) for x in most_complaints_bldg['age'].values.tolist()], width = .4, color = 'red', label = "max_compaints" )
    l2 = plt.bar(np.arange(0,10)+ 0.5, [float(x) for x in least_compaints_bldg['age'].values.tolist()], width = .4 , color = 'blue', label = "min_complaints")
    plt.axhline(y = float(cook_county_data["age"][0])/10000)
    plt.xticks(range(11))
    plt.legend(prop={'size':5})
    plt.tight_layout()
    plt.xlabel("for max complaints 10 = min, for min complaints 10 = min")
    plt.ylabel("Employment_unemployed")
    plt.title('Vacant/Abandon Complaints per Demographics (age)')
    plt.savefig("Vacant_and_Abandoned_demographics.jpeg")



def cook_county(queries = code_represents):

    result_blockwise = {}
    for key,val in code_represents.items():
        query = key
        url2 = 'http://api.census.gov/data/2015/acs5?get={}&for=county:031&in=state:17&key={}'.format(query,apikey)
        pm = urllib3.PoolManager()
        html = pm.urlopen(url=url2, method='GET').data
        soup = BeautifulSoup(html, 'lxml')
        data = soup.text
        data = data.replace('null', "'null'")
        dict_data = ast.literal_eval(data)
        list_data = dict_data
        out = list_data[1][0]
        if val not in result_blockwise:
            result_blockwise[val] = []
        result_blockwise[val].append(out)
    return result_blockwise

def url_to_dic(address):

    pm = urllib3.PoolManager()
    html = pm.urlopen(url=address, method='GET').data
    soup = BeautifulSoup(html, 'lxml')
    data = soup.text
    dict_data = ast.literal_eval(data) 
    return dict_data

def augment(req_data, queries = code_represents):

    result_blockwise = {}
    for i in range(0, len(req_data)):
    #location = (req_data_1.iloc[i]["Latitude"], req_data_1.iloc[i]['Longitude'])
        location = ast.literal_eval(req_data.index[i])
  
        url1 = 'http://data.fcc.gov/api/block/find?format=json&latitude={}&longitude={}&showall=true'.format(location[0],location[1])
        dict_data = url_to_dic(url1)
        fips = dict_data['Block']['FIPS']
        state_id = fips[:2]
        county_id = fips[2:5]
        tract_id = fips[5:11]
        block_group_id = fips[11]

        for key,val in code_represents.items():
            query = key
            url2 = 'http://api.census.gov/data/2015/acs5?get={}&for=block+group:{}&in=state:{}+county:{}+tract:{}&key={}'.format(query,block_group_id,state_id,county_id,tract_id,apikey)
            pm = urllib3.PoolManager()
            html = pm.urlopen(url=url2, method='GET').data
            soup = BeautifulSoup(html, 'lxml')
            data = soup.text
            data = data.replace('null', "'null'")
            dict_data = ast.literal_eval(data)
            list_data = dict_data
            out = list_data[1][0]
            if val not in result_blockwise:
                result_blockwise[val] = []
            result_blockwise[val].append(out)

    return result_blockwise
    

################# Question 3 ###################

def probabilities():
    
    combined_311 = pd.read_csv("Result.csv")
    
    #data for 7500 S Wolcott Ave

    lat = 41.757380
    lon = -87.671264
    ward = 17
    ward_wise_data = combined_311.loc[:,['Type of Service Request','Ward']]
    ward_clustered = pd.crosstab(ward_wise_data.loc[:,'Type of Service Request'], ward_wise_data.loc[:,'Ward'], margins=True)
    print("7500 S Wolcott Ave data\n",ward_clustered[17], "\n\n")


    lawndale = combined_311[combined_311["Street Address"].str.contains("LAWNDALE")==True]
    count_lawndale = lawndale.groupby(["Type of Service Request"]).count()["Creation Date"]
    count_uptown = ward_clustered[46] #ward 46 is for uptown
    print("Lawndale data\n", count_lawndale,"\n\n")
    print("Uptown Data\n", count_uptown)


