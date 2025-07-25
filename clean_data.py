import pandas as pd

# Load the dataset
data = pd.read_csv('gesture_data.csv')

# Clean and format the 'Landmarks' column
def clean_landmarks(landmark_string):
    try:
        # Ensure it is enclosed in square brackets and comma-separated
        landmark_string = landmark_string.strip().replace("  ", " ").replace(" ", ",")
        # Add missing brackets if necessary
        if not landmark_string.startswith("["):
            landmark_string = "[" + landmark_string
        if not landmark_string.endswith("]"):
            landmark_string = landmark_string + "]"
        return landmark_string
    except Exception as e:
        print(f"Error cleaning landmarks: {landmark_string}")
        return None

# Apply cleaning to the 'Landmarks' column
data['Landmarks'] = data['Landmarks'].apply(clean_landmarks)

# Save the cleaned data back to a new CSV file
data.dropna(subset=['Landmarks'], inplace=True)  # Drop rows with invalid landmarks
data.to_csv('cleaned_gesture_data.csv', index=False)

print("Data cleaning complete. Saved to cleaned_gesture_data.csv.")
