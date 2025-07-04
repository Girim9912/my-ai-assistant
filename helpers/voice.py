import pyttsx3
import speech_recognition as sr

engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."