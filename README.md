# Travel Itinerary Recommender  
### CS 470 – Artificial Intelligence Term Project  
**Author:** Audrey Richards  

---

## Project Overview

The Travel Itinerary Recommender is a machine learning-based desktop application that generates personalized travel recommendations based on a user's budget and preferred trip type.

The system uses the K-Nearest Neighbors (KNN) algorithm to analyze travel data and recommend the top 5 destinations that best match user preferences. Results are displayed through an interactive GUI and visualized using a scatterplot.

---

## Machine Learning Approach

This project implements the K-Nearest Neighbors (KNN) algorithm using scikit-learn.

### Features Used in the Model:
- Cost  
- Rating  
- Popularity  
- Encoded Trip Type (via LabelEncoder)

### Process:
1. Load travel data from a CSV file.
2. Encode categorical trip types into numeric values.
3. Construct a feature matrix.
4. Filter data by selected trip type.
5. Apply KNN to identify the 5 closest destinations.
6. Calculate recommendation accuracy.
7. Visualize results in a scatterplot.

---

## GUI Features

Built using Tkinter, the interface allows users to:

- Enter a travel budget
- Select a trip type (Adventurous, Relaxed, Cultural)
- Generate personalized recommendations
- View:
  - Destination
  - Cost
  - Rating
  - Recommended restaurant
  - Accommodation
  - Suggested activities
- See a visual representation of nearest neighbors

---

## Data Visualization

The program generates a scatterplot showing:

- All filtered destinations
- The 5 nearest neighbors
- The user's input point

This visualization helps demonstrate how the KNN algorithm selects recommendations.

---

## Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- Tkinter
