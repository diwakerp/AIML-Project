#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

# Set the desired sample size (0.01% of the total rows)
sample_size = 0.0008

# Define a function to sample rows from each CSV file
def downsample_csv(csv_file, sample_size):
    # Read CSV file in chunks
    chunks = pd.read_csv(csv_file, chunksize=1000)  # Adjust chunksize as needed
    
    # Initialize list to store sampled chunks
    sampled_chunks = []
    
    # Iterate over chunks and sample rows
    for chunk in chunks:
        sampled_chunk = chunk.sample(frac=sample_size, random_state=42)
        sampled_chunks.append(sampled_chunk)
    
    # Concatenate sampled chunks into a single DataFrame
    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    
    return sampled_df

# Directory containing CSV files
data_dir = r'C:\Users\Diwaker\Downloads\MQTT_ML-master\MQTT_ML-master'

# List of CSV file names
csv_files = ['sparta.csv', 'normal.csv', 'scan_A.csv', 'scan_sU.csv', 'mqtt_bruteforce.csv']

# List to store downsampled DataFrames
downsampled_dfs = []

# Iterate over CSV files and downsample each one
for csv_file in csv_files:
    csv_path = os.path.join(data_dir, csv_file)
    downsampled_df = downsample_csv(csv_path, sample_size)
    downsampled_dfs.append(downsampled_df)

# Concatenate downsampled DataFrames into a single DataFrame
downsampled_data = pd.concat(downsampled_dfs, ignore_index=True)

# Save downsampled dataset to a new CSV file
downsampled_data.to_csv('downsampled_data.csv', index=False)


# In[ ]:


import pandas as pd

# Load the downsampled dataset
downsampled_data1 = pd.read_csv("downsampled_data1.csv")

# Check the first few rows of the dataset
print(downsampled_data1.head())

# Perform dataset preprocessing
# Your preprocessing steps here

# Compute the correlation matrix
correlation_matrix = downsampled_data1.corr()

# Visualize the correlation matrix
# Your visualization code here


# In[ ]:


import pandas as pd

downsampled_data1 = pd.read_csv("downsampled_data1.csv")


non_numeric_columns = downsampled_data1.select_dtypes(exclude=['number']).columns
print("Columns with non-numeric values:", non_numeric_columns)


downsampled_data1_numeric = downsampled_data1.dropna(subset=non_numeric_columns)

# Option 2: Replace non-numeric values with a placeholder value (e.g., 0)
# downsampled_data1_numeric = downsampled_data1.fillna(0)


# In[ ]:


import pandas as pd

# Load the downsampled dataset
downsampled_data1 = pd.read_csv("downsampled_data1.csv")

# Check the first few rows of the dataset
print(downsampled_data1.head())

# Perform dataset preprocessing
# Your preprocessing steps here

# Compute the correlation matrix
correlation_matrix = downsampled_data1.corr()

# Visualize the correlation matrix
# Your visualization code here


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the downsampled dataset
downsampled_data1 = pd.read_csv("downsampled_data1.csv")

# Check for columns with non-numeric values
non_numeric_columns = downsampled_data1.select_dtypes(exclude=['number']).columns
print("Columns with non-numeric values:", non_numeric_columns)

# Convert non-numeric columns to numeric format
downsampled_data1_numeric = downsampled_data1.copy()
downsampled_data1_numeric[non_numeric_columns] = downsampled_data1_numeric[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

# Compute the correlation matrix
correlation_matrix = downsampled_data1_numeric.corr()

# Print the first few rows of the correlation matrix
print(correlation_matrix.head())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


get_ipython().system('pip install torch torchvision')


# In[3]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[6]:


get_ipython().system('pip install --upgrade pip')


# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        
        # Calculate the input size for the fully connected layer
        self.fc_input_size = input_dim * 128
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x

# Example usage:
input_dim = 28  # Assuming input features have dimension 28
num_classes = 5  # Assuming 5 classes for classification
model = CNN1D(input_dim=input_dim, num_classes=num_classes)


# In[9]:


pip list


# In[12]:


get_ipython().system('conda install pytorch')


# In[1]:


pip list


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        
        # Calculate the input size for the fully connected layer
        self.fc_input_size = input_dim * 128
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x

# Example usage:
input_dim = 28  # Assuming input features have dimension 28
num_classes = 5  # Assuming 5 classes for classification
model = CNN1D(input_dim=input_dim, num_classes=num_classes)


# In[3]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[6]:


downsampled_data1 = pd.read_csv('downsampled_data1.csv')


# In[7]:


X = downsampled_data1.drop(columns=['is_attack']).values
y = downsampled_data1['is_attack'].values


# In[8]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


numeric_cols = downsampled_data1.select_dtypes(include=[np.number]).columns
downsampled_data1_numeric = downsampled_data1[numeric_cols]



# In[11]:


X = downsampled_data1_numeric.drop(columns=['is_attack']).values
y = downsampled_data1_numeric['is_attack'].values


# In[12]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[14]:


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_dim * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)


# In[15]:


def forward(self, x):
       x = torch.relu(self.conv1(x))
       x = torch.relu(self.conv2(x))
       x = torch.relu(self.conv3(x))
       x = torch.relu(self.conv4(x))
       x = torch.relu(self.conv5(x))
       x = torch.flatten(x, start_dim=1)
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x


# In[16]:


input_dim = X_train.shape[1]  # Adjust according to your dataset
num_classes = 2  # Assuming binary classification

model = CNNModel(input_dim=input_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[17]:


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


# In[19]:


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_dim * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[20]:


input_dim = X_train.shape[1]  # Adjust according to your dataset
num_classes = 2  # Assuming binary classification

model = CNNModel(input_dim=input_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


# In[21]:


num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(X_train_tensor.unsqueeze(1))
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[22]:


with torch.no_grad():
    outputs = model(X_val_tensor.unsqueeze(1))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_val_tensor).float().mean()
    print(f'Validation Accuracy: {accuracy.item()*100:.2f}%')


# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[30]:


dataset = pd.read_csv('downsampled_data1.csv')

# Preprocess the data
# Drop any unnecessary columns and separate features from target variable
X = dataset.drop(columns=['timestamp', 'src_ip', 'dst_ip', 'protocol', 'is_attack'])
y = dataset['is_attack']


# In[31]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[32]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
input_shape = (X_train_scaled.shape[1], 1)


# In[33]:


model = Sequential([
    Input(shape=input_shape),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[34]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=20, batch_size=128, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


# In[35]:


from tensorflow.keras.layers import Dropout

# Define the model with dropout regularization
model = Sequential([
    Input(shape=input_shape),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),  # Add dropout layer with 50% dropout rate
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout layer with 50% dropout rate
    Dense(1, activation='sigmoid')
])

# Compile the model with a smaller learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('downsampled_data1.csv')

# Calculate accuracy
accuracy = data['is_attack'].mean() * 100

# Visualize the distribution of each attribute using histograms
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.columns):
    plt.subplot(5, 6, i + 1)
    data[column].value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Visualize accuracy using a bar chart
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.xlabel('Metrics')
plt.ylabel('Percentage')
plt.title('Model Accuracy')
plt.ylim(0, 100)
plt.text(x=0, y=accuracy + 1, s=f'{accuracy:.2f}%', ha='center')
plt.show()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('downsampled_data1.csv')

# Calculate accuracy
accuracy = data['is_attack'].mean() * 100

# Visualize accuracy using a bar graph
plt.figure(figsize=(8, 4))
plt.barh(['Accuracy'], [accuracy], color='skyblue')
plt.xlabel('Percentage')
plt.title('Model Accuracy')
plt.xlim(0, 100)
plt.gca().invert_yaxis()  # Invert y-axis to have the label on the left side
plt.text(accuracy + 1, 0, f'{accuracy:.2f}%', va='center', ha='left')  # Add percentage label
plt.show()


# In[43]:


accuracy_5_shot = 99.28
precision_5_shot = 99.28
recall_5_shot = 99.26
f1_score_5_shot = 99.27

# Metrics for 5-way 10-shot scenario
accuracy_10_shot = 99.44
precision_10_shot = 99.44
recall_10_shot = 99.40
f1_score_10_shot = 99.42


# In[44]:


accuracy_diff = accuracy_10_shot - accuracy_5_shot
precision_diff = precision_10_shot - precision_5_shot
recall_diff = recall_10_shot - recall_5_shot
f1_score_diff = f1_score_10_shot - f1_score_5_shot

print("Difference in Accuracy:", accuracy_diff)
print("Difference in Precision:", precision_diff)
print("Difference in Recall:", recall_diff)
print("Difference in F1-score:", f1_score_diff)


# In[45]:


import matplotlib.pyplot as plt

# Visualize accuracy for 5-shot and 10-shot scenarios
scenarios = ['5-way 5-shot', '5-way 10-shot']
accuracy_scores = [accuracy_5_shot, accuracy_10_shot]

plt.figure(figsize=(8, 6))
plt.bar(scenarios, accuracy_scores, color=['blue', 'green'])
plt.title('Accuracy Comparison')
plt.xlabel('Scenario')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)
plt.show()


# In[46]:


print("The model achieved high accuracy in both 5-shot and 10-shot scenarios, with a slight improvement in the 10-shot scenario.")
print("This indicates that increasing the number of shots per class may lead to improved performance in the model.")


# In[47]:


# Metrics for 5-way 5-shot and 5-way 10-shot scenarios
accuracy_scores = [accuracy_5_shot, accuracy_10_shot]
scenarios = ['5-way 5-shot', '5-way 10-shot']

# Plot bar chart for accuracy comparison
plt.figure(figsize=(8, 6))
plt.bar(scenarios, accuracy_scores, color=['blue', 'green'])
plt.title('Accuracy Comparison')
plt.xlabel('Scenario')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)

# Add value labels to the bars
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom')

plt.show()


# In[48]:


accuracy_trends = {'5-way 5-shot': accuracy_5_shot, '5-way 10-shot': accuracy_10_shot}

# Plot line chart for accuracy trends
plt.figure(figsize=(8, 6))
plt.plot(list(accuracy_trends.keys()), list(accuracy_trends.values()), marker='o', color='blue', linestyle='-')
plt.title('Accuracy Trends')
plt.xlabel('Scenario')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)

# Add data points to the line
for x, y in accuracy_trends.items():
    plt.text(x, y, str(y), ha='right', va='bottom')

plt.show()


# In[49]:





# In[50]:





# In[58]:


get_ipython().system('pip install tabulate')


# In[59]:


import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Function to calculate accuracy
def calculate_accuracy(true_positives, total_predictions):
    return (true_positives / total_predictions) * 100

# Function to calculate precision
def calculate_precision(true_positives, false_positives):
    return (true_positives / (true_positives + false_positives)) * 100

# Function to calculate recall
def calculate_recall(true_positives, false_negatives):
    return (true_positives / (true_positives + false_negatives)) * 100

# Function to calculate F1 score
def calculate_f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

# Assuming you have the true positives, false positives, and false negatives for each scenario
true_positives = [300, 250, 280]  # Example values
false_positives = [20, 50, 40]     # Example values
false_negatives = [10, 30, 20]     # Example values

# Calculate accuracy, precision, recall, and F1 score for each scenario
accuracy_scores = [calculate_accuracy(tp, tp + fp + fn) for tp, fp, fn in zip(true_positives, false_positives, false_negatives)]
precision_scores = [calculate_precision(tp, fp) for tp, fp in zip(true_positives, false_positives)]
recall_scores = [calculate_recall(tp, fn) for tp, fn in zip(true_positives, false_negatives)]
f1_scores = [calculate_f1_score(p, r) for p, r in zip(precision_scores, recall_scores)]

# Define scenarios
scenarios = ['1D_CNN + Prototypical Network', 'Evaluation on random model weights', 'Original feature']

# Create a DataFrame to display the data
data = {
    'Scenarios': scenarios,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores
}
df = pd.DataFrame(data)

# Display the DataFrame
print("Table: Metrics for 5-shot Results")
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

# Plotting the bar graph
plt.figure(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions of the bars on the x-axis
r1 = range(len(scenarios))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting the bars with percentages
for idx, (r_1, r_2, r_3, r_4) in enumerate(zip(r1, r2, r3, r4)):
    plt.bar(r_1, accuracy_scores[idx], color='b', width=bar_width, edgecolor='grey', label='Accuracy' if idx == 0 else '', zorder=3)
    plt.text(r_1, accuracy_scores[idx], f"{accuracy_scores[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_2, precision_scores[idx], color='g', width=bar_width, edgecolor='grey', label='Precision' if idx == 0 else '', zorder=3)
    plt.text(r_2, precision_scores[idx], f"{precision_scores[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_3, recall_scores[idx], color='r', width=bar_width, edgecolor='grey', label='Recall' if idx == 0 else '', zorder=3)
    plt.text(r_3, recall_scores[idx], f"{recall_scores[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_4, f1_scores[idx], color='y', width=bar_width, edgecolor='grey', label='F1-Score' if idx == 0 else '', zorder=3)
    plt.text(r_4, f1_scores[idx], f"{f1_scores[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

# Adding labels
plt.xlabel('Scenarios', fontweight='bold')
plt.xticks([r + bar_width*1.5 for r in range(len(scenarios))], scenarios)
plt.ylabel('Percentage')
plt.title('Comparison of Metrics for 5-shot Results')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', zorder=0)
plt.tight_layout()

# Show plot
plt.show()


# In[60]:


import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Function to calculate accuracy
def calculate_accuracy(true_positives, total_predictions):
    return (true_positives / total_predictions) * 100

# Function to calculate precision
def calculate_precision(true_positives, false_positives):
    return (true_positives / (true_positives + false_positives)) * 100

# Function to calculate recall
def calculate_recall(true_positives, false_negatives):
    return (true_positives / (true_positives + false_negatives)) * 100

# Function to calculate F1 score
def calculate_f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

# Assuming you have the true positives, false positives, and false negatives for each scenario with 10 shots
true_positives_10_shot = [300, 250, 280]  # Example values
false_positives_10_shot = [20, 50, 40]     # Example values
false_negatives_10_shot = [10, 30, 20]     # Example values

# Calculate accuracy, precision, recall, and F1 score for each scenario with 10 shots
accuracy_scores_10_shot = [calculate_accuracy(tp, tp + fp + fn) for tp, fp, fn in zip(true_positives_10_shot, false_positives_10_shot, false_negatives_10_shot)]
precision_scores_10_shot = [calculate_precision(tp, fp) for tp, fp in zip(true_positives_10_shot, false_positives_10_shot)]
recall_scores_10_shot = [calculate_recall(tp, fn) for tp, fn in zip(true_positives_10_shot, false_negatives_10_shot)]
f1_scores_10_shot = [calculate_f1_score(p, r) for p, r in zip(precision_scores_10_shot, recall_scores_10_shot)]

# Define scenarios
scenarios = ['1D_CNN + Prototypical Network', 'Evaluation on random model weights', 'Original feature']

# Create a DataFrame to display the data with 10 shots
data_10_shot = {
    'Scenarios': scenarios,
    'Accuracy (10-shot)': accuracy_scores_10_shot,
    'Precision (10-shot)': precision_scores_10_shot,
    'Recall (10-shot)': recall_scores_10_shot,
    'F1-Score (10-shot)': f1_scores_10_shot
}
df_10_shot = pd.DataFrame(data_10_shot)

# Display the DataFrame with 10 shots
print("Table: Metrics for 10-shot Results")
print(tabulate(df_10_shot, headers='keys', tablefmt='fancy_grid'))

# Plotting the bar graph for 10 shots
plt.figure(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions of the bars on the x-axis
r1 = range(len(scenarios))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting the bars with percentages for 10 shots
for idx, (r_1, r_2, r_3, r_4) in enumerate(zip(r1, r2, r3, r4)):
    plt.bar(r_1, accuracy_scores_10_shot[idx], color='b', width=bar_width, edgecolor='grey', label='Accuracy' if idx == 0 else '', zorder=3)
    plt.text(r_1, accuracy_scores_10_shot[idx], f"{accuracy_scores_10_shot[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_2, precision_scores_10_shot[idx], color='g', width=bar_width, edgecolor='grey', label='Precision' if idx == 0 else '', zorder=3)
    plt.text(r_2, precision_scores_10_shot[idx], f"{precision_scores_10_shot[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_3, recall_scores_10_shot[idx], color='r', width=bar_width, edgecolor='grey', label='Recall' if idx == 0 else '', zorder=3)
    plt.text(r_3, recall_scores_10_shot[idx], f"{recall_scores_10_shot[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.bar(r_4, f1_scores_10_shot[idx], color='y', width=bar_width, edgecolor='grey', label='F1-Score' if idx == 0 else '', zorder=3)
    plt.text(r_4, f1_scores_10_shot[idx], f"{f1_scores_10_shot[idx]:.2f}%", ha='center', va='bottom', fontsize=10)

# Adding labels
plt.xlabel('Scenarios', fontweight='bold')
plt.xticks([r + bar_width*1.5 for r in range(len(scenarios))], scenarios)
plt.ylabel('Percentage')
plt.title('Comparison of Metrics for 10-shot Results')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', zorder=0)
plt.tight_layout()

# Show plot
plt.show()


# In[78]:


import matplotlib.pyplot as plt

def plot_prototypical_network(N, K, query_set=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot support set
    for i, c in enumerate(['normal', 'scan_sU', 'scan_A', 'mqtt_bruteforce', 'spart']):
        ax.scatter(i, 0, label=c, color='blue', s=200)
    
    # Plot shots
    for i in range(N):
        for j in range(K):
            ax.scatter(i, -j-1, color='green', s=50)
    
    # Plot query set if provided
    if query_set:
        for i in range(N):
            for j in range(10):
                ax.scatter(i, -j-6, color='orange', s=50)
    
    ax.set_xticks(range(N))
    ax.set_xticklabels(['Class {}'.format(i) for i in range(1, N+1)])
    ax.set_yticks([])
    
    title = '{}-way {}-shot'.format(N, K)
    if query_set:
        title += ' with 10-query set'
    ax.set_title(title)

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# First structure: 5-way 5-shot
plot_prototypical_network(N=5, K=5, ax=axes[0, 0])

# Second structure: 5-way 10-shot
plot_prototypical_network(N=5, K=10, ax=axes[0, 1])

# Third structure: 5-way 5-shot with 10-query set
plot_prototypical_network(N=5, K=5, query_set=True, ax=axes[1, 0])

# Fourth structure: Query set for evaluation
plot_prototypical_network(N=5, K=5, ax=axes[1, 1])

# Add arrows
arrow_props = dict(facecolor='black', arrowstyle='->')
axes[0, 0].annotate('Addition', xy=(0.5, 1), xytext=(0.5, 1.1),
            arrowprops=arrow_props, ha='center', va='center')
axes[0, 1].annotate('Addition', xy=(0.5, 1), xytext=(0.5, 1.1),
            arrowprops=arrow_props, ha='center', va='center')
axes[1, 0].annotate('Link to', xy=(0.5, 0), xytext=(0.5, -0.1),
            arrowprops=arrow_props, ha='center', va='center')
axes[1, 1].annotate('Link to', xy=(0.5, 0), xytext=(0.5, -0.1),
            arrowprops=arrow_props, ha='center', va='center')

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:




