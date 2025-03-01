import gradio as gr
import os
from groq import Groq

# Groq client initialization

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# conversation history initialization
conversation_history = []

# Function to handle the chat conversation with streaming
def chat_with_bot_stream(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) == 1:
        conversation_history.insert(0, {
            "role": "system",
            "content": "You are a friendly chef. Respond with detailed recipes for any dish the user asks for."
        })

    completion = client.chat.completions.create(
        model="llama3-70b-8192", messages=conversation_history,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": response_content})

    return [(msg["content"] if msg["role"] == "user" else None, 
             msg["content"] if msg["role"] == "assistant" else None) 
            for msg in conversation_history]

# Create the Gradio interface with tabs
TITLE = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>🍳 Recipe Assistant</h1>
"""

with gr.Blocks(theme=gr.themes.Glass(primary_hue="orange", secondary_hue="yellow", neutral_hue="stone")) as demo:
    with gr.Tabs():
        # Tab 1: Chatbot
        with gr.TabItem("💬 Chat with Chef"):
            gr.HTML(TITLE)
            chatbot = gr.Chatbot(label="Recipe Chatbot")
            with gr.Row():
                user_input = gr.Textbox(label="Ask for a recipe", placeholder="Type your question here...", lines=1)
                send_button = gr.Button("Ask Recipe")
            
            # Chatbot functionality
            send_button.click(
                fn=chat_with_bot_stream,
                inputs=user_input,
                outputs=chatbot,
                queue=True
            ).then(
                fn=lambda: "", 
                inputs=None, 
                outputs=user_input
            )


demo.launch()
