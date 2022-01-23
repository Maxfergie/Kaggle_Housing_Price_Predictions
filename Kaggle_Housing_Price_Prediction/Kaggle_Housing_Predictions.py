# %% [markdown]
# **This notebook is an exercise in the [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/pipelines).**
# 
# ---
# 

# %% [markdown]
# In this exercise, you will use **pipelines** to improve the efficiency of your machine learning code.
# 
# # Setup
# 
# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:15.075331Z","iopub.execute_input":"2021-12-09T18:23:15.075851Z","iopub.status.idle":"2021-12-09T18:23:15.08713Z","shell.execute_reply.started":"2021-12-09T18:23:15.075817Z","shell.execute_reply":"2021-12-09T18:23:15.086147Z"}}
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex4 import *
print("Setup Complete")

# %% [markdown]
# You will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and validation sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  The test set is loaded in `X_test`.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:15.088775Z","iopub.execute_input":"2021-12-09T18:23:15.089192Z","iopub.status.idle":"2021-12-09T18:23:15.176692Z","shell.execute_reply.started":"2021-12-09T18:23:15.089161Z","shell.execute_reply":"2021-12-09T18:23:15.175797Z"}}
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, 
                                                                y, 
                                                                train_size=0.8, 
                                                                test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:15.177887Z","iopub.execute_input":"2021-12-09T18:23:15.178114Z","iopub.status.idle":"2021-12-09T18:23:15.200534Z","shell.execute_reply.started":"2021-12-09T18:23:15.178086Z","shell.execute_reply":"2021-12-09T18:23:15.199666Z"}}
X_train.head()

# %% [markdown]
# The next code cell uses code from the tutorial to preprocess the data and train a model.  Run this code without changes.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:15.201748Z","iopub.execute_input":"2021-12-09T18:23:15.202001Z","iopub.status.idle":"2021-12-09T18:23:17.821556Z","shell.execute_reply.started":"2021-12-09T18:23:15.201971Z","shell.execute_reply":"2021-12-09T18:23:17.820707Z"}}
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

# %% [markdown]
# The code yields a value around 17862 for the mean absolute error (MAE).  In the next step, you will amend the code to do better.
# 
# # Step 1: Improve the performance
# 
# ### Part A
# 
# Now, it's your turn!  In the code cell below, define your own preprocessing steps and random forest model.  Fill in values for the following variables:
# - `numerical_transformer`
# - `categorical_transformer`
# - `model`
# 
# To pass this part of the exercise, you need only define valid preprocessing steps and a random forest model.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:17.824406Z","iopub.execute_input":"2021-12-09T18:23:17.82481Z","iopub.status.idle":"2021-12-09T18:23:17.835185Z","shell.execute_reply.started":"2021-12-09T18:23:17.824762Z","shell.execute_reply":"2021-12-09T18:23:17.834396Z"}}
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant') # Your code here

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0) # Your code here

# Check your answer
step_1.a.check()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:17.836408Z","iopub.execute_input":"2021-12-09T18:23:17.83664Z","iopub.status.idle":"2021-12-09T18:23:17.845213Z","shell.execute_reply.started":"2021-12-09T18:23:17.836613Z","shell.execute_reply":"2021-12-09T18:23:17.844518Z"}}
# Lines below will give you a hint or solution code
#step_1.a.hint()
#step_1.a.solution()

# %% [markdown]
# ### Part B
# 
# Run the code cell below without changes.
# 
# To pass this step, you need to have defined a pipeline in **Part A** that achieves lower MAE than the code above.  You're encouraged to take your time here and try out many different approaches, to see how low you can get the MAE!  (_If your code does not pass, please amend the preprocessing steps and model in Part A._)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:17.846662Z","iopub.execute_input":"2021-12-09T18:23:17.847507Z","iopub.status.idle":"2021-12-09T18:23:20.611454Z","shell.execute_reply.started":"2021-12-09T18:23:17.847461Z","shell.execute_reply":"2021-12-09T18:23:20.610464Z"}}
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# Check your answer
step_1.b.check()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:20.613024Z","iopub.execute_input":"2021-12-09T18:23:20.613343Z","iopub.status.idle":"2021-12-09T18:23:20.617174Z","shell.execute_reply.started":"2021-12-09T18:23:20.61331Z","shell.execute_reply":"2021-12-09T18:23:20.615934Z"}}
# Line below will give you a hint
#step_1.b.hint()

# %% [markdown]
# # Step 2: Generate test predictions
# 
# Now, you'll use your trained model to generate predictions with the test data.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:20.61886Z","iopub.execute_input":"2021-12-09T18:23:20.619153Z","iopub.status.idle":"2021-12-09T18:23:20.710179Z","shell.execute_reply.started":"2021-12-09T18:23:20.619114Z","shell.execute_reply":"2021-12-09T18:23:20.709224Z"}}
# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)
 # Your code here

# Check your answer
step_2.check()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:20.711794Z","iopub.execute_input":"2021-12-09T18:23:20.712031Z","iopub.status.idle":"2021-12-09T18:23:20.71505Z","shell.execute_reply.started":"2021-12-09T18:23:20.711976Z","shell.execute_reply":"2021-12-09T18:23:20.714468Z"}}
# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()

# %% [markdown]
# Run the next code cell without changes to save your results to a CSV file that can be submitted directly to the competition.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-09T18:23:20.716333Z","iopub.execute_input":"2021-12-09T18:23:20.716665Z","iopub.status.idle":"2021-12-09T18:23:20.735155Z","shell.execute_reply.started":"2021-12-09T18:23:20.716622Z","shell.execute_reply":"2021-12-09T18:23:20.734466Z"}}
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

# %% [markdown]
# # Submit your results
# 
# Once you have successfully completed Step 2, you're ready to submit your results to the leaderboard!  If you choose to do so, make sure that you have already joined the competition by clicking on the **Join Competition** button at [this link](https://www.kaggle.com/c/home-data-for-ml-course).  
# 1. Begin by clicking on the **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  
# 2. Ensure that the **Save and Run All** option is selected, and then click on the **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the file you would like to submit, and click on the **Submit** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# If you want to keep working to improve your performance, select the **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
# 
# 
# # Keep going
# 
# Move on to learn about [**cross-validation**](https://www.kaggle.com/alexisbcook/cross-validation), a technique you can use to obtain more accurate estimates of model performance!

# %% [markdown]
# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/intermediate-machine-learning/discussion) to chat with other learners.*