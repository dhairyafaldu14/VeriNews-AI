
import streamlit as st
import joblib

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="VeriNews AI",
    page_icon="📰",
    layout="centered"
)

# ----------------------------------
# Load Trained Model and Vectorizer
# ----------------------------------

@st.cache_resource
def load_models():
    tfidf = joblib.load('verinews_tfidf.pkl')
    model = joblib.load('verinews_svm_model.pkl')
    return tfidf, model


tfidf, model = load_models()


# ----------------------------------
# Application Heading
# ----------------------------------

st.title("📰 VeriNews AI")

st.subheader("Intelligent News Classification System")

st.write(
    "Analyze news articles using a Machine Learning "
    "model trained on the WELFake dataset."
)


# ----------------------------------
# User Input
# ----------------------------------

title = st.text_input(
    "News Headline",
    placeholder="Enter the news headline..."
)

article = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the complete news article here..."
)


# ----------------------------------
# News Classification
# ----------------------------------

if st.button("Classify News", type="primary"):

    if title.strip() and article.strip():

        # Combine headline and article
        content = title.strip() + " " + article.strip()

        # Convert text into TF-IDF features
        article_tfidf = tfidf.transform([content])

        # Make prediction
        prediction = model.predict(article_tfidf)[0]

        # Calculate SVM decision score
        score = model.decision_function(article_tfidf)[0]


        # ----------------------------------
        # Classification Result
        # ----------------------------------

        st.divider()

        st.header("Classification Result")

        if prediction == 1:

            st.error("🚨 Prediction: Fake News")

            category = "Fake News"

            explanation = (
                "The model's decision score is positive, "
                "so it classified the input as Fake News."
            )

        else:

            st.success("✅ Prediction: Real News")

            category = "Real News"

            explanation = (
                "The model's decision score is negative, "
                "so it classified the input as Real News."
            )


        # Display prediction details
        st.metric(
            label="Predicted Category",
            value=category
        )

        st.metric(
            label="SVM Decision Score",
            value=f"{score:.4f}"
        )


        # ----------------------------------
        # Prediction Explanation
        # ----------------------------------

        st.subheader("Why This Prediction?")

        st.write(explanation)

        st.info(
            "A positive decision score indicates Fake News, "
            "while a negative score indicates Real News. "
            "The absolute score measures distance from the "
            "decision boundary, not a probability."
        )


        # ----------------------------------
        # Influential Word Analysis
        # ----------------------------------

        st.subheader("Important Influential Words")

        feature_names = tfidf.get_feature_names_out()

        # Calculate each feature's contribution
        contributions = (
            article_tfidf.multiply(model.coef_[0])
            .toarray()
            .flatten()
        )

        # Select the top five contributing features
        if prediction == 1:

            top_indices = contributions.argsort()[-5:][::-1]

            st.write(
                "Words and phrases contributing most "
                "toward the Fake classification:"
            )

        else:

            top_indices = contributions.argsort()[:5]

            st.write(
                "Words and phrases contributing most "
                "toward the Real classification:"
            )


        # Display influential words
        displayed_words = 0

        for index in top_indices:

            contribution = contributions[index]

            if contribution != 0:

                st.write(
                    f"- **{feature_names[index]}** "
                    f"({contribution:.4f})"
                )

                displayed_words += 1


        if displayed_words == 0:

            st.write(
                "No individual word contributions "
                "were available for display."
            )


        # ----------------------------------
        # Disclaimer
        # ----------------------------------

        st.warning(
            "This prediction is based on patterns learned "
            "from the WELFake dataset. The influential words "
            "show model contributions, not evidence of "
            "factual accuracy. This application does not "
            "independently verify news articles."
        )


    else:

        st.warning(
            "Please enter both the headline and news article."
        )


# ----------------------------------
# Footer
# ----------------------------------

st.divider()

st.caption(
    "VeriNews AI | Machine Learning News Classification"
)
