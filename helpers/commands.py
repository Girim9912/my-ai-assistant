import webbrowser

def run_command(text):
    text = text.lower()
    if "open google" in text:
        webbrowser.open("https://google.com")
        return "Opening Google."
    elif "open youtube" in text:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube."
    elif "exit" in text:
        return "Exit command detected. Please close the app manually."
    return None