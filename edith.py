from flask import Flask, render_template, jsonify
import speech_recognition as sr
import pyttsx3
import webbrowser
import wikipedia
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ----------------
# SETUP
# ----------------
app = Flask(__name__)
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Training data for intents
commands = [
    "open google", "search on google",
    "open youtube", "play music on youtube",
    "wikipedia machine learning", "tell me about python wikipedia",
    "send email", "compose an email",
    "what is the weather", "get live weather"
]
intents = [
    "google", "google",
    "youtube", "youtube",
    "wikipedia", "wikipedia",
    "email", "email",
    "weather", "weather"
]

# Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(commands)

# Models
nb_model = MultinomialNB().fit(X, intents)
dt_model = DecisionTreeClassifier().fit(X, intents)
lr_model = LogisticRegression(max_iter=1000).fit(X, intents)

def classify_intent(command):
    X_test = vectorizer.transform([command])
    # Majority voting between models
    nb_pred = nb_model.predict(X_test)[0]
    dt_pred = dt_model.predict(X_test)[0]
    lr_pred = lr_model.predict(X_test)[0]

    predictions = [nb_pred, dt_pred, lr_pred]
    return max(set(predictions), key=predictions.count)

def process_command(command):
    intent = classify_intent(command.lower())
    response = "Sorry, I didn’t understand."

    if intent == "google":
        webbrowser.open("https://google.com")
        response = "Opening Google."
    elif intent == "youtube":
        webbrowser.open("https://youtube.com")
        response = "Opening YouTube."
    elif intent == "wikipedia":
        try:
            topic = command.replace("wikipedia", "").strip()
            summary = wikipedia.summary(topic, sentences=1)
            response = summary
        except:
            response = "Could not fetch Wikipedia results."
    elif intent == "email":
        response = "Email feature not yet implemented."
    elif intent == "weather":
        response = "Fetching weather (dummy response)."
    return intent, response

# ----------------
# ROUTES
# ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/listen")
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source, timeout=5)

        command = recognizer.recognize_google(audio)
        print("User said:", command)

        # 🔥 Call your ML function here
        intent = predict_intent(command)   # <-- you already have this
        response = f"Intent detected: {intent}"

        engine.say(response)
        engine.runAndWait()

        return jsonify({"command": command, "intent": intent, "response": response})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
