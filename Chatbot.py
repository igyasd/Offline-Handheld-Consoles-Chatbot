from sentence_transformers import SentenceTransformer,util
import tkinter as tk
from tkinter import scrolledtext

model= SentenceTransformer('all-MiniLM-L6-v2')

qa_pairs = [
    ("Hi there!", "Hi! How can I help with handheld gaming consoles today?"),
    ("Hello!", "Hey! Looking for advice on handheld consoles?"),
    ("How are you?", "I'm here and ready to talk gaming handhelds!"),

    ("What’s a good handheld console for emulation?", "You might like an Android handheld like Retroid Pocket or Ayn Odin for emulation and flexibility."),
    ("Which handheld is best for retro gaming?", "Retro handhelds like Anbernic RG35XX or Miyoo Mini are excellent for classic games."),
    ("Can handhelds emulate PS2 games?", "Yes, stronger handhelds like Steam Deck or Ayn Odin can handle many PS2 titles."),
    ("Can I emulate GameCube games on a handheld?", "Steam Deck and Ayn Odin are popular choices for GameCube emulation."),
    ("Can I emulate PSP games?", "Most modern handhelds can run PSP games very well."),

    ("Which handheld can run PC games?", "Steam Deck and ASUS ROG Ally are designed for PC gaming on the go."),
    ("What is the most powerful handheld console?", "The ROG Ally and Steam Deck OLED are among the strongest gaming handhelds available."),
    ("Can I play AAA games on a handheld?", "Yes, devices like Steam Deck and ROG Ally can run many AAA games."),
    ("Can I install Steam on a handheld?", "Steam comes built into Steam Deck, and can also be installed on Windows-based handhelds."),
    ("Can I play Xbox games on a handheld?", "You can use Xbox Cloud Gaming or remote play on supported devices."),
    ("Can I play PlayStation games on a handheld?", "Yes, remote play apps and emulators can work depending on the device."),

    ("Is the Nintendo Switch worth it?", "Yes! The Switch is excellent for Nintendo exclusives and flexible play styles."),
    ("What games are popular on Switch?", "Mario Kart, Zelda, Animal Crossing, Smash Bros, and Pokémon are popular choices."),
    ("Can I connect the Switch to a TV?", "Yes, most Switch models support TV output through a dock."),
    ("Is Switch OLED better?", "The OLED model mainly improves the screen, kickstand, and audio experience."),

    ("What’s a good cheap handheld console?", "Budget-friendly options include Anbernic and Powkiddy handhelds."),
    ("I only have a small budget", "Retro handhelds usually offer the best value at lower prices."),
    ("What’s the cheapest handheld for emulation?", "Entry-level Anbernic devices are often affordable choices."),

    ("Which handheld has the best battery life?", "Retro handhelds generally last longer because they use less power."),
    ("How long does Steam Deck battery last?", "Battery life depends on the game, but demanding games drain it faster."),
    ("How long does Switch battery last?", "Battery life varies, but most sessions last several hours."),

    ("Can I play online multiplayer on handheld consoles?", "Yes, Switch and PC handhelds support online gaming."),
    ("Can I use Wi-Fi on handheld consoles?", "Most modern handhelds include Wi-Fi support."),
    ("Do handheld consoles support Bluetooth?", "Many modern handhelds support Bluetooth for headphones and controllers."),

    ("Can I connect a controller?", "Yes, many handhelds support external controllers."),
    ("Can I use headphones with a handheld?", "Yes, wired and Bluetooth headphones are commonly supported."),
    ("Can I use a keyboard and mouse?", "PC-based handhelds often support keyboard and mouse connections."),

    ("Which handheld is best for traveling?", "Smaller devices like Miyoo Mini are easier to carry while traveling."),
    ("What’s the smallest handheld console?", "Pocket-sized devices like Miyoo Mini or RG Nano are very compact."),
    ("Which handheld has a large screen?", "Steam Deck OLED and Logitech G Cloud offer larger displays."),

    ("How much storage do I need?", "64–256GB works for casual use, while larger libraries benefit from more storage."),
    ("Can I expand storage?", "Many handhelds support microSD cards for additional storage."),
    ("What accessories should I buy?", "Useful accessories include a case, screen protector, dock, and microSD card."),

    ("Which handheld should I buy?", "It depends on whether you want PC gaming, Nintendo titles, or retro emulation."),
    ("What is the best handheld overall?", "There isn’t one answer — the best choice depends on your needs."),
    ("Thanks!", "You're welcome! Feel free to ask more about handheld gaming."),
]


question_texts = [q for q, a in qa_pairs]
question_embeddings = model.encode(
    question_texts,
    convert_to_tensor=True
)

# Similarity threshold — below this, the bot says it doesn't understand
THRESHOLD = 0.3


# Semantic matching replaces keyword matching
def get_response(user_input):
    input_embedding = model.encode(
        user_input,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        input_embedding,
        question_embeddings
    )[0]

    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    if best_score < THRESHOLD:
        return (
            best_score,
            "Sorry, I don't understand. Try asking about pasta, dessert, or vegan recipes!"
        )

    return best_score, qa_pairs[best_idx][1]



class ChatbotUI:
    def __init__(self, win):
        self.win = win
        self.win.title("Recipe Chatbot")
        self.win.geometry("500x600")
        self.win.configure(bg="#2E2E2E")

        # Title
        tk.Label(
            win, text="Recipe Chatbot", font=("Helvetica", 16, "bold"),
            fg="#FFFFFF", bg="#2E2E2E"
        ).pack(pady=10)

        # Chat area (scrollable)
        self.chat_area = scrolledtext.ScrolledText(
            win, wrap=tk.WORD, height=20, width=50, font=("Arial", 11),
            bg="#3C3C3C", fg="#E0E0E0", insertbackground="white"
        )
        self.chat_area.pack(pady=10, padx=10)
        self.chat_area.insert(tk.END,
                              "Welcome to the Handheld Chatbot!\n"
                              "Ask Questions about Handhelds (e.g., 'Best emulation Handheld').\n")
        self.chat_area.config(state='disabled')

        # Input frame
        input_frame = tk.Frame(win, bg="#2E2E2E")
        input_frame.pack(pady=5)

        # Input field
        self.input_field = tk.Entry(
            input_frame, width=40, font=("Arial", 11),
            bg="#4A4A4A", fg="#FFFFFF",
            insertbackground="white"
        )
        self.input_field.pack(side=tk.LEFT, padx=5)
        self.input_field.bind("<Return>", self.send_message)

        # Send button
        tk.Button(
            input_frame, text="Send", command=self.send_message,
            font=("Arial", 11),
            bg="#4CAF50", fg="#FFFFFF",
            activebackground="#45A049"
        ).pack(side=tk.LEFT, padx=5)

        # Clear button
        tk.Button(
            win, text="Clear Chat", command=self.clear_chat,
            font=("Arial", 11),
            bg="#F44336", fg="#FFFFFF",
            activebackground="#D32F2F"
        ).pack(pady=5)

    def send_message(self, event=None):
        user_input = self.input_field.get().strip()
        if not user_input:
            return

        score, response = get_response(user_input)
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"\nYou: {user_input}\n")
        self.chat_area.insert(tk.END, f"Match confidence: {score:.2f}\n")
        self.chat_area.insert(tk.END, f"Bot: {response}\n")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)

        self.input_field.delete(0, tk.END)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)

        self.chat_area.insert(
            tk.END,
            "Welcome to the Recipe Chatbot!\n"
            "Ask about recipes (e.g., 'something sweet' or 'healthy lunch ideas').\n"
        )

        self.chat_area.config(state='disabled')


def main():
    win=tk.Tk()
    app=ChatbotUI(win)
    win.mainloop()

if __name__=="__main__":
    main()