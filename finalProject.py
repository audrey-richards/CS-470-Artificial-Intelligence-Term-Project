# Audrey Richards
# CS 470 Term Project
# This project takes a csv file with a destination, TripType, Cost, Rating, Popularity,
# Restaurants, Accomodations, and Activities for this location. The user enter a budget
# and selects the desired trip type, and this program outputs 5 recomended locations,
# with a cost, rating, and reccomended resturant, accommodation, and trip
# -------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from tkinter import ttk

# DATA PREPARATION --------------------------------------------------------------------------------

# Load travel data from CSV file.
input_file = 'travel_data.csv'
data = pd.read_csv(input_file)

# Store the relevant columns into lists
# .values allows you to access the column and converts it
# from a pd array to a np array
destinations = data["City"].values
countries = data["Country"].values
trip_type = data["TripType"].values
cost = data["Cost"].values
rating = data["Rating"].values
popularity = data["Popularity"].values
restaurants = data["Restaurants"].values
accommodations = data["Accommodations"].values
activities = data["Activities"].values

# use label encoder to convert categorical columns (TripType) to numbers
label_encoder = LabelEncoder()
encoded_trip_type = label_encoder.fit_transform(trip_type)

# testing label encoding
"""
print("\nEncoded label mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
"""

# Combine numeric features into a feature matrix
features = np.column_stack((cost, rating, popularity, encoded_trip_type))

# REFERENCES FOR DATA PREPARATION -----------------------------------------------------------------
#
# how I learned to convert pd array to np array and access each column
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html
#
# how I learned to combine numueric features into a feature matrix
# https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html

# RECOMMENDATION FUNCTION USING K NEAREST NEIGHBORS ALGORITHM -------------------------------------

def recommend_travel():
    # Get user input from the GUI
    user_budget = float(entry_budget.get()) # convert input to float 
    user_trip_type = trip_type_var.get() # get selected trip type from dropdown

    # validate the trip type
    if user_trip_type not in ["Adventurous", "Relaxed", "Cultural"]:
        listbox_results.insert(END, "Invalid trip type. Please select from the dropdown.")
        return
    
    # Encode user's trip type selection
    user_trip_type_encoded = label_encoder.transform([user_trip_type])[0]
    
    # Filter destinations matching the user's trip type
    filtered_data = data[data["TripType"] == user_trip_type]
    
    # create a filtered feature matrix for k nearest neighbor processing
    filtered_features = np.column_stack((
        # Combind numeric features from the filtered data
        filtered_data["Cost"].values,
        filtered_data["Rating"].values,
        filtered_data["Popularity"].values,
        label_encoder.transform(filtered_data["TripType"].values)
    ))
    
    # input array based on the user's budget and the average rating/popularity of filtered data
    user_input = np.array([[user_budget, filtered_data["Rating"].mean(),
                            filtered_data["Popularity"].mean(), user_trip_type_encoded]])
    
    # Apply K Nearest Neighbors on filtered data to find the 5 closest matches
    k = min(5, len(filtered_data)) # make sure we don't exceed 5 neighbors
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(filtered_features)
    distances, indices = knn_model.kneighbors(user_input)
    
    # Intialize accuracy variables
    correct_matches = 0
    total_recommendations = len(indices[0])
    
    # Loop through recommendation indicies and display the results
    listbox_results.delete(0, END) # Clear any previous results
    for i in indices[0]:
        recommended_row = filtered_data.iloc[i] # get the data row for the corresponding recommendation
        destination = recommended_row["City"]
        country = recommended_row["Country"]
        
        # Check if the recommendation matches the user's preferences for accuracy calculation
        if (recommended_row["TripType"] == user_trip_type and recommended_row["Cost"] <= user_budget):
            correct_matches += 1
        
        # display recommendation details
        listbox_results.insert(END, f"Destination: " + destination + ", " + country)
        listbox_results.insert(END, f"  Cost: $" + str(recommended_row['Cost']))
        listbox_results.insert(END, f"  Rating: " + str(recommended_row['Rating']))
        listbox_results.insert(END, f"  Recommended Restaurant: " + recommended_row['Restaurants'])
        listbox_results.insert(END, f"  Recommended Accommodation: " + recommended_row['Accommodations'])
        listbox_results.insert(END, f"  Recommended trip: " + recommended_row['Activities'])
        listbox_results.insert(END, "") # Empty line between destinations
    
    # Calculate and display accuracy
    if total_recommendations > 0:
        accuracy = (correct_matches / total_recommendations) * 100
    else: 0
    listbox_results.insert(END, f"Recommendation Accuracy: " + str(accuracy))
    
    # Plot the nearest neighbors with the user's input
    plt.figure()
    plt.title('Nearest neighbors')
    
    # Scatter plot for the filtered destinations
    plt.scatter(filtered_features[:, 0], filtered_features[:, 3], marker='o', s=50, color='lightblue', label = 'Travel Desintations')
    # Highlight the nearest neighbors
    plt.scatter(filtered_features[indices[0], 0], filtered_features[indices[0], 3], 
        marker='o', s=150, color='black', facecolors='none', label='Nearest Neighbors')
    # Mark the user's input point
    plt.scatter(user_input[:, 0], user_input[:, 3], marker = 'x', s = 200, color='red', label='Your Input')
    
    # Set y-axis labels to show trip types instead of encoded values
    plt.yticks(
        ticks=[0, 1, 2],
        labels=label_encoder.inverse_transform([0, 1, 2])
    )
    
    plt.xlabel('Cost')
    plt.ylabel('Trip Type')
    plt.legend()
    # Show the graph
    plt.show()

# REFERENCES FOR RECOMMENDATION SYSTEM ---------------------------------------------------------------------
#
# This Code came from lesson 12 file - k_nearest_neighor.py
#
# Endcoded data code from lesson 10b clustering - traffic_prediction.py
#
# How to find the data row for the corresponding reccomendation
#    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
#
# How to compute the accuracy for the reccomendation system
# code idea came from lesson 11 file - gmm_classifier_fixed.py
#

# GUI INTERFACE --------------------------------------------------------------------------------------------

# Create the main Tkinter window for the GUI
window = Tk()
window.title("Travel Itinerary Recommender")
window.geometry("600x500")
window["bg"] = "LightBlue"

# Create title label
title = Label(window, text = "Travel Itinerary Recommender", fg="black", bg="white", font=("Comfortaa", 16))
title.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="W")

# Create label and entry for budget
label_budget = Label(window, text="Enter your budget ($): ", fg="black", bg="white", font=("Comfortaa", 12))
label_budget.grid(row=1, column=0, padx=10, pady=5, sticky="W")
entry_budget = Entry(window, fg="black", bg="white", width=15)
entry_budget.grid(row=1, column=1, padx=10, pady=5, sticky="W")

# Create label and dropdown for trip type
label_trip = Label(window, text="Select your preferred trip type: ", fg="black", bg="white", font=("Comfortaa", 12))
label_trip.grid(row=2, column=0, padx=10, pady=5, sticky="W")
trip_type_var = StringVar()
dropdown_trip = ttk.Combobox(window, textvariable=trip_type_var, state="readonly")
dropdown_trip['values'] = ["Adventurous", "Relaxed", "Cultural"]
dropdown_trip.grid(row=2, column=1, padx=10, pady=5, sticky="W")
dropdown_trip.current(0) # Set default to 'Adventure'

# Create button to generate itinerary
button_generate = Button(window, text="Generate Itinerary", fg="black", bg="white", command=recommend_travel, font=("Comfortaa", 12))
button_generate.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="WE")

# Create listbox to display recommnedations
listbox_results = Listbox(window, bg="white", fg="black", height=12, width=50, font=("Comfortaa", 14))
listbox_results.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="WE")

# Create scrollbar for listbox
scrollbar = Scrollbar(window, command=listbox_results.yview)
scrollbar.grid(row=4, column=2, sticky="NS")
listbox_results.config(yscrollcommand=scrollbar.set)

# Start the Tkinter loop
window.mainloop()

# REFERENCES FOR GUI INTERFACE ------------------------------------------------------------------------------------------
#
# learned how to create GUI system through tkinter
# https://www.geeksforgeeks.org/create-first-gui-application-using-python-tkinter/
#
# learned how to use listbox
# https://www.geeksforgeeks.org/python-tkinter-listbox-widget/
#
# learned how to implement scrollbar
# https://www.geeksforgeeks.org/python-tkinter-scrollbar/
#
# learned how to implement selection drop down
# https://www.geeksforgeeks.org/combobox-widget-in-tkinter-python/
