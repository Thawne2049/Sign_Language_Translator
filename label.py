from sklearn.preprocessing import LabelEncoder
import pickle

# List of categories used in your old work (adjust if needed)
categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize LabelEncoder and fit it to the categories
label_encoder = LabelEncoder()
label_encoder.fit(categories)

# Save the LabelEncoder as a pickle file
with open('label_encoder_isl.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("LabelEncoder has been recreated and saved.")


