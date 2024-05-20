
import tkinter as tk
from tkinter import scrolledtext
from nltk.chat.util import Chat, reflections

# Travel Agent responses
Travel_Agent_Responses = [
    (r"Hello", ["Hi, How may i assist you today?"]),
    (r"Wanted to know about Tour Packages", ["We have customised packages for Beach Destinations & Hill Stations."]),
    (r"Ok", ["Please provide us your requirements."]),
    (r"Sure will do", ["We'll get back to you with the itinerary details via email."]),
    (r"Thanks", ["Bye! Have a Nice Day."])
]

# Create a chatbot
chatbot = Chat(Travel_Agent_Responses, reflections)

# Function to handle sending a message and getting a response
def send_message():
    message = user_input.get()
    user_input.delete(0, tk.END)
    response = chatbot.respond(message)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "You: " + message + "\n", "user")
    chat_history.insert(tk.END, "Bot: " + response + "\n", "bot")
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)

# Create the main window
root = tk.Tk()
root.title("Travel Agent Chatbot")

# Create widgets
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD)
user_input = tk.Entry(root)
send_button = tk.Button(root, text="Send", command=send_message)

# Configure chat history
chat_history.config(state=tk.DISABLED)
chat_history.tag_config("user", foreground="blue")
chat_history.tag_config("bot", foreground="red")

# Place widgets on the grid
chat_history.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
user_input.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
send_button.grid(row=1, column=1, padx=5, pady=5, sticky="e")

# Start the main loop
root.mainloop()
