# app.py - Personal Assistant using Hugging Face models
import streamlit as st
import pyttsx3
import webbrowser
import threading
import time
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if torch.cuda.is_available() else "CPU"

# Initialize TTS engine
@st.cache_resource
def get_tts_engine():
    try:
        engine = pyttsx3.init()
        return engine
    except:
        return None

# Load different models based on user choice
@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "DialoGPT":
            return pipeline("text-generation", 
                          model="microsoft/DialoGPT-medium",
                          device=device,
                          pad_token_id=50256)
        
        elif model_name == "BlenderBot":
            return pipeline("text2text-generation",
                          model="facebook/blenderbot-400M-distill",
                          device=device)
        
        elif model_name == "Flan-T5":
            return pipeline("text2text-generation",
                          model="google/flan-t5-small",
                          device=device)
        
        elif model_name == "GPT-2":
            return pipeline("text-generation",
                          model="gpt2",
                          device=device,
                          pad_token_id=50256)
        
        elif model_name == "DistilGPT2":
            return pipeline("text-generation",
                          model="distilgpt2",
                          device=device,
                          pad_token_id=50256)
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Smart command trigger words
command_keywords = {
    "open google": lambda: webbrowser.open("https://google.com"),
    "open youtube": lambda: webbrowser.open("https://youtube.com"),
    "open github": lambda: webbrowser.open("https://github.com"),
    "open stackoverflow": lambda: webbrowser.open("https://stackoverflow.com"),
    "open reddit": lambda: webbrowser.open("https://reddit.com"),
    "open twitter": lambda: webbrowser.open("https://twitter.com"),
    "open huggingface": lambda: webbrowser.open("https://huggingface.co"),
    "what time": lambda: f"The current time is {datetime.now().strftime('%I:%M %p')}",
    "what date": lambda: f"Today's date is {datetime.now().strftime('%B %d, %Y')}",
}

def smart_command_check(user_input):
    for keyword in command_keywords:
        if keyword in user_input.lower():
            if keyword in ["what time", "what date"]:
                return command_keywords[keyword]()
            else:
                command_keywords[keyword]()
                return f"✅ Opened {keyword.replace('open ', '')}"
    return None

def generate_response(model, model_name, user_input, conversation_history):
    """Generate response using the selected model"""
    try:
        if model_name == "DialoGPT":
            # Build conversation context
            context = ""
            if conversation_history:
                recent_history = conversation_history[-4:]  # Last 4 messages
                for msg in recent_history:
                    if msg.startswith("You:"):
                        context += f"Human: {msg[4:].strip()}\n"
                    else:
                        context += f"Assistant: {msg[10:].strip()}\n"
            
            prompt = f"{context}Human: {user_input}\nAssistant:"
            
            result = model(prompt, 
                         max_length=len(prompt.split()) + 50,
                         do_sample=True,
                         temperature=0.7,
                         top_p=0.9,
                         repetition_penalty=1.1)
            
            response = result[0]['generated_text']
            response = response.split("Assistant:")[-1].strip()
            response = response.split("Human:")[0].strip()
            
            return response if response else "I'm thinking... can you rephrase that?"
        
        elif model_name == "BlenderBot":
            result = model(user_input, max_length=100, do_sample=True, temperature=0.7)
            return result[0]['generated_text']
        
        elif model_name == "Flan-T5":
            prompt = f"You are a helpful personal assistant. Respond to: {user_input}"
            result = model(prompt, max_length=100, do_sample=True, temperature=0.7)
            return result[0]['generated_text']
        
        elif model_name in ["GPT-2", "DistilGPT2"]:
            prompt = f"Human: {user_input}\nAssistant:"
            result = model(prompt, 
                         max_length=len(prompt.split()) + 40,
                         do_sample=True,
                         temperature=0.7,
                         top_p=0.9,
                         repetition_penalty=1.1)
            
            response = result[0]['generated_text']
            response = response.split("Assistant:")[-1].strip()
            response = response.split("Human:")[0].strip()
            
            return response if response else "Let me think about that..."
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def speak_text(text):
    """Function to handle TTS in a separate thread"""
    try:
        engine = get_tts_engine()
        if engine:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {e}")

def main():
    st.set_page_config(page_title="HuggingFace AI Assistant", page_icon="🤗")
    
    st.title("🤗 HuggingFace Personal AI Assistant")
    st.write(f"Powered by Hugging Face Transformers • Running on {device_name}")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'speech_enabled' not in st.session_state:
        st.session_state.speech_enabled = False
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "DialoGPT"

    # Model selection
    st.subheader("🧠 Choose Your AI Model")
    model_options = {
        "DialoGPT": "Microsoft DialoGPT (Best for conversations)",
        "BlenderBot": "Facebook BlenderBot (Conversational AI)",
        "Flan-T5": "Google Flan-T5 (Instruction following)",
        "GPT-2": "OpenAI GPT-2 (Creative text generation)",
        "DistilGPT2": "DistilGPT2 (Lightweight & Fast)"
    }
    
    selected_model = st.selectbox(
        "Select Model:", 
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # Load model if changed
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.session_state.current_model = None
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.current_model = load_model(selected_model)
        if st.session_state.current_model:
            st.success(f"✅ {selected_model} loaded successfully!")
        else:
            st.error(f"❌ Failed to load {selected_model}")
    
    # Load model if not loaded
    if st.session_state.current_model is None:
        with st.spinner(f"Loading {st.session_state.model_name}..."):
            st.session_state.current_model = load_model(st.session_state.model_name)

    # User input
    user_input = st.text_input("You:", key="input", placeholder="Ask me anything or give me a command...")

    # Buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        send_button = st.button("Send 🚀")
    with col2:
        clear_button = st.button("Clear 🗑️")
    with col3:
        speech_toggle = st.button("Speech 🔊")
    with col4:
        help_button = st.button("Help ❓")

    if speech_toggle:
        st.session_state.speech_enabled = not st.session_state.speech_enabled
        status = "enabled" if st.session_state.speech_enabled else "disabled"
        st.info(f"🔊 Speech {status}")

    if help_button:
        st.info("💡 Try: 'open google', 'what time', 'tell me a joke', or ask anything!")

    if clear_button:
        st.session_state.conversation_history = []
        st.success("🧹 Conversation history cleared!")
        st.rerun()

    if send_button and user_input and st.session_state.current_model:
        # Check if it's a command
        command_response = smart_command_check(user_input)
        if command_response:
            st.success(f"🤖 {command_response}")
            st.session_state.conversation_history.append(f"You: {user_input}")
            st.session_state.conversation_history.append(f"Assistant: {command_response}")
            
            if st.session_state.speech_enabled:
                threading.Thread(target=speak_text, args=(command_response,), daemon=True).start()
        else:
            # Generate AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = generate_response(
                        st.session_state.current_model,
                        st.session_state.model_name,
                        user_input,
                        st.session_state.conversation_history
                    )
                    
                    st.session_state.conversation_history.append(f"You: {user_input}")
                    st.session_state.conversation_history.append(f"Assistant: {response}")
                    
                    st.success(f"🤖 {response}")
                    
                    if st.session_state.speech_enabled:
                        threading.Thread(target=speak_text, args=(response,), daemon=True).start()
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    fallback_response = "I'm having trouble processing that. Can you try rephrasing?"
                    st.session_state.conversation_history.append(f"You: {user_input}")
                    st.session_state.conversation_history.append(f"Assistant: {fallback_response}")
                    
                    if st.session_state.speech_enabled:
                        threading.Thread(target=speak_text, args=(fallback_response,), daemon=True).start()

    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        for message in reversed(st.session_state.conversation_history[-10:]):
            if message.startswith("You:"):
                st.write(f"🧑 {message}")
            else:
                st.write(f"🤖 {message}")

    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Commands")
        commands = [
            "open google", "open youtube", "open github", 
            "open stackoverflow", "open huggingface",
            "what time", "what date"
        ]
        for cmd in commands:
            st.write(f"• '{cmd}'")
        
        st.header("🔧 Model Info")
        st.write(f"**Current:** {st.session_state.model_name}")
        st.write(f"**Device:** {device_name}")
        
        st.header("💡 Tips")
        st.write("• DialoGPT: Best for conversations")
        st.write("• BlenderBot: Great personality")
        st.write("• Flan-T5: Follows instructions well")
        st.write("• GPT-2: Creative responses")
        st.write("• DistilGPT2: Fastest option")
        
        speech_status = "🔊 ON" if st.session_state.speech_enabled else "🔇 OFF"
        st.write(f"**Speech:** {speech_status}")

if __name__ == "__main__":
    main()