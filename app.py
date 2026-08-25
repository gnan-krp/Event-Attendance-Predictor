import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Event Attendance Predictor",
    page_icon="🎯",
    layout="wide"
)

model = joblib.load("attendance_model.pkl")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #888;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.15);
        margin-top: 15px;
    }

    .small-note {
        font-size: 13px;
        color: #999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🎯 Event Attendance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict how likely a registered student is to actually attend a club event.</div>',
    unsafe_allow_html=True
)

st.info(
    "💡 Enter the registration details below and the model will estimate the attendance probability."
)

left, right = st.columns(2)

with left:

    st.subheader("🎟️ Event Details")

    event_type = st.selectbox(
        "Event Type",
        [
            "Workshop",
            "Hackathon",
            "Talk",
            "Competition",
            "Social"
        ]
    )

    event_day = st.selectbox(
        "Event Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    event_hour = st.slider(
        "Event Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    registration_days_before = st.number_input(
        "Registration Days Before Event",
        min_value=0,
        value=5,
        step=1
    )

with right:

    st.subheader("👤 Student Details")

    previous_events_registered = st.number_input(
        "Previous Events Registered",
        min_value=0,
        value=0,
        step=1
    )

    previous_events_attended = st.number_input(
        "Previous Events Attended",
        min_value=0,
        value=0,
        step=1
    )

    club_member = st.selectbox(
        "Club Member",
        [
            "Yes",
            "No"
        ]
    )

    travel_distance_km = st.number_input(
        "Travel Distance (km)",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

st.divider()

if st.button(
    "🚀 Predict Attendance",
    use_container_width=True,
    type="primary"
):

    corrected_registered = max(
        previous_events_registered,
        previous_events_attended
    )

    if corrected_registered == 0:
        previous_attendance_rate = np.nan
    else:
        previous_attendance_rate = (
            previous_events_attended
            / corrected_registered
        )

    input_data = pd.DataFrame(
        {
            "event_type": [
                event_type.lower()
            ],
            "registration_days_before": [
                registration_days_before
            ],
            "previous_events_registered": [
                corrected_registered
            ],
            "previous_events_attended": [
                previous_events_attended
            ],
            "club_member": [
                club_member.lower()
            ],
            "event_day": [
                event_day.lower()
            ],
            "travel_distance_km": [
                travel_distance_km
            ],
            "event_hour": [
                event_hour
            ],
            "previous_attendance_rate": [
                previous_attendance_rate
            ]
        }
    )

    probability = model.predict_proba(
        input_data
    )[0][1]

    percentage = probability * 100

    st.subheader("📊 Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Attendance Probability",
            f"{percentage:.1f}%"
        )

    with metric2:
        st.metric(
            "Previous Attendance Rate",
            (
                "No History"
                if np.isnan(previous_attendance_rate)
                else f"{previous_attendance_rate * 100:.1f}%"
            )
        )

    with metric3:
        st.metric(
            "Registration Timing",
            f"{registration_days_before} days early"
        )

    st.progress(
        int(percentage)
    )

    if probability >= 0.75:

        st.success(
            f"🟢 Strong chance of attendance — {percentage:.1f}%"
        )

        st.write(
            "This student looks like a highly reliable attendee based on the historical patterns learned by the model."
        )

    elif probability >= 0.50:

        st.warning(
            f"🟡 Moderate chance of attendance — {percentage:.1f}%"
        )

        st.write(
            "The student is more likely to attend than not, but a reminder may still be useful."
        )

    else:

        st.error(
            f"🔴 High no-show risk — {percentage:.1f}% attendance probability"
        )

        st.write(
            "This registration may benefit from additional reminders or engagement."
        )

    st.divider()

    st.subheader("🧠 Quick Interpretation")

    if registration_days_before >= 8:
        st.write(
            "✅ Early registration may be a positive signal."
        )
    elif registration_days_before <= 2:
        st.write(
            "⚠️ Last-minute registration may indicate higher no-show risk."
        )

    if club_member == "Yes":
        st.write(
            "✅ Existing club membership is associated with stronger attendance in the historical data."
        )
    else:
        st.write(
            "💬 Non-members may benefit from extra follow-up before the event."
        )

    if event_type == "Workshop":
        st.write(
            "🛠️ Workshops showed the strongest observed attendance in the historical dataset."
        )

    st.caption(
        "These insights are based on historical patterns and should not be interpreted as guaranteed causal effects."
    )

st.divider()

with st.expander("📌 About the Model"):

    st.write(
        """
        This app uses a Gradient Boosting Classifier trained on historical
        event registration data.

        The model considers:

        - Event type
        - Registration timing
        - Previous registrations
        - Previous attendance
        - Club membership
        - Event day
        - Event hour
        - Travel distance
        - Previous attendance rate
        """
    )

    st.write(
        """
        Validation Results:

        Precision: 72.46%

        Recall: 78.12%

        F1-score: 75.19%

        ROC-AUC: 68.83%
        """
    )

st.caption(
    "🤖 Event Attendance Predictor | Powered by Gradient Boosting"
)