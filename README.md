# 🎯 Event Attendance Predictor

> **Will a registered student actually show up?** 🤔

Every club faces the same problem.

You announce an event.  
100 students register. 🎉  
You prepare everything expecting 100 people...

...and then only 60 actually show up. 😭

So I thought — **can Machine Learning help predict this?**

This project uses historical event registration data to predict how likely a student is to actually attend an event after registering.

---

# 🚀 What Does This Project Do?

The **Event Attendance Predictor** is a binary classification Machine Learning model that analyzes different factors related to a student's registration and estimates their probability of attending the event.

Instead of simply giving:

```text
Attend / Not Attend
```

the model gives a probability:

```text
👨‍🎓 Student S1379 → 88.5% likely to attend
👨‍🎓 Student S1273 → 29.0% likely to attend
👨‍🎓 Student S1015 → 96.2% likely to attend
```

This could help clubs estimate actual turnout and plan events more efficiently.

I also created an interactive **Streamlit application** where registration details can be entered manually and the trained model generates an attendance probability instantly. 🎨

---

# 💡 Why This Project?

Event registrations don't always represent actual attendance.

A club may receive **200 registrations**, but that doesn't necessarily mean 200 students will show up.

This creates problems while planning:

- 🍕 Food and refreshments
- 🪑 Seating arrangements
- 🏫 Venue capacity
- 🎁 Swag and goodies
- 👥 Volunteer requirements
- 💰 Event budget
- 📢 Reminder campaigns

Instead of assuming:

> **Registrations = Attendance**

this project tries to estimate:

> **How likely is each registered student to actually attend?**

---

# 📊 Dataset

The project uses two datasets:

### 🏋️ Training Dataset

Contains historical registrations **along with the actual attendance result**.

```text
508 registrations
10 columns
```

### 🧪 Test Dataset

Contains new registrations for which the model needs to predict attendance.

```text
100 registrations
9 columns
```

The difference is that the training dataset contains:

```text
attended
```

while the test dataset does not.

That's exactly what the model needs to predict. 🎯

---

# 🧩 Features Used

The model uses several pieces of information about each registration.

| Feature | What it represents |
|---|---|
| 🎟️ `event_type` | Type of event such as workshop, hackathon, talk, etc. |
| 📅 `registration_days_before` | How early the student registered |
| 🔢 `previous_events_registered` | Number of previous events registered for |
| ✅ `previous_events_attended` | Number of previous events actually attended |
| 👥 `club_member` | Whether the student is already a club member |
| 🗓️ `event_day` | Day on which the event is happening |
| 🕐 `event_time` | Time of the event |
| 📍 `travel_distance_km` | Approximate travel distance to the event |

The training dataset additionally contains:

```text
attended
```

which is the **target variable**.

---

# 🧹 Data Cleaning

Real-world data is rarely perfectly clean.

Before training the model, I handled several issues in the dataset.

### 🔤 Inconsistent Categorical Values

Values such as:

```text
Workshop
workshop
WORKSHOP
```

should obviously represent the same thing.

So categorical values are cleaned using:

```python
value.strip().lower()
```

which converts them into a consistent format:

```text
workshop
```

---

### 🕳️ Missing Values

Missing numerical values are filled using the **median**:

```python
SimpleImputer(strategy="median")
```

Missing categorical values are filled using the **most frequent value**:

```python
SimpleImputer(strategy="most_frequent")
```

This allows the model to train without simply deleting registrations containing incomplete information.

---

# 🛠️ Feature Engineering

I also created additional features from the existing data to give the model more useful information.

## 🕐 Event Hour

Instead of directly using values such as:

```text
18:30
```

the hour is extracted:

```text
18:30 → 18
```

creating:

```text
event_hour
```

This gives the model a simple numerical representation of when the event takes place.

---

## 📈 Previous Attendance Rate

Simply knowing how many events someone attended doesn't tell the complete story.

Consider:

```text
Student A
Registered: 2
Attended:   2

Student B
Registered: 20
Attended:   10
```

Student B attended more events overall, but Student A has a much stronger attendance history.

So I created:

```text
previous_attendance_rate
```

using:

```text
previous_events_attended
--------------------------
previous_events_registered
```

Example:

```text
8 attended / 10 registered
            ↓
           0.80
            ↓
      80% attendance rate
```

---

# 🔢 Handling Categorical Data

Machine Learning models work with numbers, not words like:

```text
workshop
hackathon
social
```

So categorical variables are converted using:

### 🔥 One-Hot Encoding

For example:

```text
event_type = workshop
```

could become:

```text
workshop   hackathon   social
   1           0          0
```

This is handled using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

---

# ⚙️ ML Pipeline

Instead of manually preprocessing every dataset separately, the project uses a Scikit-learn **Pipeline**.

The overall flow looks like this:

```text
                📊 Raw Registration Data
                          │
                          ▼
                    🧹 Data Cleaning
                          │
                          ▼
                   🛠️ Feature Engineering
                          │
                          ▼
              ┌────────────────────────┐
              │     Preprocessing      │
              └────────────────────────┘
                    │            │
                    ▼            ▼
              🔢 Numerical    🔤 Categorical
                    │            │
                 Median       Most Frequent
                 Imputer        Imputer
                                  │
                                  ▼
                            One-Hot Encoding
                    │            │
                    └──────┬─────┘
                           ▼
                   🤖 Gradient Boosting
                           │
                           ▼
                  🎯 Attendance Probability
```

---

# 🤖 Model Used

## Gradient Boosting Classifier 🌳🌳🌳

The final model uses:

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)
```

Gradient Boosting builds multiple small decision trees.

Each new tree tries to improve the mistakes made by the previous trees.

Very roughly:

```text
🌳 Tree 1
   ↓
Makes some mistakes
   ↓
🌳 Tree 2
   ↓
Focuses on those mistakes
   ↓
🌳 Tree 3
   ↓
Improves further
   ↓
...
   ↓
🌳 Tree 200
   ↓
🎯 Final Prediction
```

Gradient Boosting works particularly well with structured/tabular data and can learn nonlinear relationships between different features.

---

# 🧪 Train / Validation Split

The labeled dataset is divided into:

```text
80% → 🏋️ Training
20% → 🧪 Validation
```

using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The model learns using the training portion.

The validation portion contains data the model did **not see during training**, giving a more realistic idea of its performance.

---

# 📏 Model Evaluation

I evaluated the model using:

- 🎯 Precision
- 🔎 Recall
- ⚖️ F1-score
- 📈 ROC-AUC
- 🔢 Confusion Matrix

The validation results were:

| Metric | Score |
|---|---:|
| 🎯 **Precision** | **72.46%** |
| 🔎 **Recall** | **78.12%** |
| ⚖️ **F1-score** | **75.19%** |
| 📈 **ROC-AUC** | **68.83%** |

---

# 🧠 What Do These Metrics Mean?

### 🎯 Precision — 72.46%

Of all the students predicted to attend, approximately **72% actually attended** in the validation data.

### 🔎 Recall — 78.12%

Of all the students who actually attended, the model successfully identified approximately **78% of them**.

For this project, recall is particularly useful because we want to identify as many potential attendees as possible.

### ⚖️ F1-score — 75.19%

F1-score balances **Precision and Recall**.

It provides a useful overall measure when both false positives and false negatives matter.

---

# 🔢 Confusion Matrix

The model produced:

```text
[[18 19]
 [14 50]]
```

Which means:

| | Count |
|---|---:|
| ✅ True Positive | 50 |
| ✅ True Negative | 18 |
| ⚠️ False Positive | 19 |
| ⚠️ False Negative | 14 |

So the model correctly identified **50 actual attendees** in the validation set.

---

# 🎯 Attendance Probability

The model doesn't only output `0` or `1`.

It calculates:

```python
model.predict_proba()
```

which gives the estimated probability of attendance.

For example:

```text
S1379 → 88.5%
S1425 → 83.9%
S1273 → 29.0%
S1015 → 96.2%
S1422 → 15.0%
```

A default threshold of **50%** is used:

```text
Probability ≥ 50%
        ↓
🟢 Likely to Attend

Probability < 50%
        ↓
🔴 Unlikely to Attend
```

The threshold could also be adjusted depending on what the club wants to prioritize.

---

# 🔮 Example Predictions

Some predictions generated by the model:

| Student | Attendance Probability | Prediction |
|---|---:|---|
| S1015 | 🔥 **96.2%** | 🟢 Likely to Attend |
| S1055 | **92.7%** | 🟢 Likely to Attend |
| S1475 | **90.8%** | 🟢 Likely to Attend |
| S1379 | **88.5%** | 🟢 Likely to Attend |
| S1273 | **29.0%** | 🔴 Unlikely to Attend |
| S1422 | **15.0%** | 🔴 Unlikely to Attend |
| S1278 | **13.0%** | 🔴 Unlikely to Attend |

The final predictions are saved automatically to:

```text
attendance_predictions.xlsx
```

---

# 🎨 Interactive Streamlit App

I also wanted to make the model easier to actually use instead of only running predictions through Python.

So I built a small **Streamlit interface**. 🚀

The user can enter:

```text
🎟️ Event Type
📅 Registration Timing
📝 Previous Events Registered
✅ Previous Events Attended
👥 Club Membership
🗓️ Event Day
🕐 Event Hour
📍 Travel Distance
```

and then click:

```text
🚀 Predict Attendance
```

The application passes those values to the trained Machine Learning pipeline and generates the student's estimated attendance probability.

The basic flow is:

```text
        👤 Student Registration Details
                     │
                     ▼
              🎨 Streamlit App
                     │
                     ▼
          🤖 Saved Trained Model
                     │
                     ▼
           📊 Attendance Probability
                     │
             ┌───────┴───────┐
             ▼               ▼
      🟢 Likely          🔴 No-Show Risk
```

The app also shows a visual probability bar and a quick interpretation of the prediction.

---

# 💾 Saving the Trained Model

After validation, the model is trained again using the complete training dataset.

The complete Scikit-learn pipeline is then saved using **Joblib**:

```python
joblib.dump(
    model,
    "attendance_model.pkl"
)
```

This creates:

```text
attendance_model.pkl
```

The Streamlit application loads this model using:

```python
model = joblib.load("attendance_model.pkl")
```

This means the app doesn't need to retrain the model every time someone wants to make a prediction.

---

# 💡 What Did I Learn From the Data?

The project isn't useful only for predictions.

Analyzing historical attendance also revealed some interesting patterns. 👀

## ⏰ 1. Early Registrations = Better Attendance

Students registering earlier were noticeably more likely to attend.

| Registration Time | Attendance |
|---|---:|
| 🔴 0–2 days before | **46.2%** |
| 🟡 3–7 days before | **58.4%** |
| 🟢 8–14 days before | **72.8%** |

### 💡 Possible Action

Encourage students to register earlier and consider sending stronger reminders to last-minute registrants.

---

## 👥 2. Club Members Are More Reliable

Attendance among club members:

```text
69.5%
```

Attendance among non-members:

```text
51.1%
```

That's roughly an **18 percentage-point difference**.

### 💡 Possible Action

Non-members could receive additional engagement after registering:

```text
📩 Personalized reminder
🎤 Speaker information
🗓️ Event schedule
🔥 Event highlights
📍 Venue reminder
```

---

## 🛠️ 3. Workshops Had the Highest Attendance

Event format also showed an interesting pattern.

| Event | Attendance |
|---|---:|
| 🛠️ **Workshop** | **75.4%** |
| 🎤 Talk | **66.0%** |
| 🏆 Competition | **58.5%** |
| 💻 Hackathon | **52.9%** |
| 🎉 Social | **52.6%** |

Hands-on workshops had the highest observed attendance.

### 💡 Possible Action

The club could organize more practical workshops or add hands-on elements to other event formats.

---

# ⚠️ Correlation ≠ Causation

These findings are **patterns observed in the dataset**.

For example:

> Students who register earlier were more likely to attend.

does **not necessarily mean**:

> Forcing someone to register earlier will cause them to attend.

There may be other factors involved.

The insights should therefore be used to guide decisions rather than treated as guaranteed causal relationships.

---

# 🗂️ Project Structure

```text
Event-Attendance-Predictor/
│
├── 📄 event_attendance_predictor.py
│
├── 🎨 app.py
│
├── 🤖 attendance_model.pkl
│
├── 📊 event_attendance_real_world.xlsx
│
├── 🧪 event_attendance_test_real_world.xlsx
│
├── 🎯 attendance_predictions.xlsx
│
├── 📝 README.md
└── ⚙️ .gitignore
```

---

# 💻 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/gnan-krp/Event-Attendance-Predictor.git
```

Move inside the project:

```bash
cd Event-Attendance-Predictor
```

---

## 2️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn openpyxl streamlit joblib
```

---

## 4️⃣ Train the Model 🚀

Run:

```bash
python3 event_attendance_predictor.py
```

The program will:

```text
📂 Load datasets
       ↓
🧹 Clean data
       ↓
🛠️ Engineer features
       ↓
🤖 Train Gradient Boosting
       ↓
📏 Evaluate the model
       ↓
🧠 Train final model
       ↓
💾 Save attendance_model.pkl
       ↓
🎯 Generate probabilities
       ↓
📊 Save predictions
```

After running successfully, two important files are generated:

```text
attendance_model.pkl
attendance_predictions.xlsx
```

---

# 🎨 Run the Streamlit App

After the trained model is available, run:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in your browser.

You can then enter student and event information and click:

```text
🚀 Predict Attendance
```

to receive an estimated attendance probability.

The Streamlit application currently runs **locally**, so no online deployment is required.

To stop the Streamlit server:

```text
Control + C
```

---

# 🧰 Tech Stack

### 🐍 Python

Main programming language.

### 🐼 Pandas

Dataset loading, cleaning and manipulation.

### 🔢 NumPy

Numerical operations and missing-value handling.

### 🤖 Scikit-learn

Used for:

```text
Preprocessing
One-Hot Encoding
Imputation
Pipeline
Gradient Boosting
Model Evaluation
```

### 📗 OpenPyXL

Used by Pandas for reading and writing Excel files.

### 🎨 Streamlit

Used to build the interactive interface where users can enter registration details and receive attendance predictions.

### 💾 Joblib

Used to save and load the complete trained Machine Learning pipeline.

---

# 🚧 Limitations

No ML model can perfectly predict human behaviour.

Someone might have a **95% predicted attendance probability** and still not show up because of:

```text
🌧️ Weather
📚 Exams
🤒 Illness
🚗 Transport problems
⏰ Last-minute schedule changes
😴 Or... they simply changed their mind 😭
```

The training dataset is also relatively small, with around **500 historical registrations**.

So predictions should be interpreted as:

> **Model-estimated attendance likelihood**

and not as guaranteed outcomes.

---

# 🔮 Future Improvements

There are a lot of directions in which this project could be extended.

### 🌦️ Add External Factors

Features such as:

```text
Weather
Exam schedule
Event duration
Venue
Speaker popularity
Food availability
```

could potentially improve predictions.

### 📩 Smart Reminder System

Instead of treating everyone equally:

```text
90% probability → Normal reminder

55% probability → Strong reminder

25% probability → Personalized follow-up
```

### 📊 Club Dashboard

The Streamlit application could eventually be expanded into a complete club dashboard:

```text
Registrations:            150
Expected Attendance:      ~104
High Confidence:           72
Medium Confidence:         39
High No-Show Risk:         39
```

### 🧠 Compare More Models

Future versions could compare:

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost
LightGBM
```

and tune their hyperparameters.

### 🌐 Online Deployment

The Streamlit application currently runs locally.

A future version could be deployed online so club organizers can access the predictor directly through a browser.

---

# 🌟 Bigger Idea

The interesting part of this project isn't just:

> **“Can we predict who will attend?”**

The bigger question is:

> **“Can clubs use historical behaviour to organize smarter events?”**

If attendance can be estimated beforehand, clubs could potentially make better decisions about:

```text
🍕 Food
🏫 Venues
👥 Volunteers
💰 Budgets
📢 Reminders
🎁 Resources
```

That's where a simple ML prediction can become something practically useful.

---

# 👨‍💻 Author

**Gnan Parekh**

Built while exploring Machine Learning and trying to solve a very real club-event problem. 🚀

If you found the project interesting, feel free to ⭐ the repository!

---

### ⭐ Event Attendance Predictor

**Registrations tell us who *wants* to come.**

**Machine Learning helps us estimate who might actually show up.** 🎯
