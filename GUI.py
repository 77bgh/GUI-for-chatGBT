# https://github.com/77bgh
import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
import tkinter as tk
from threading import Thread
from functools import partial
import gc

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int
    reset: bool


class ChatbotGUI:
    def __init__(self):
        self.question_history = []
        self.doc_question_history = []
        self.answer_history = []

        self.config = AutoConfig.from_pretrained(
            os.path.abspath("models"),
            context_length=6048,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
            model_type="replit",
            config=self.config,
        )

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0,
            seed=42,
            reset=True,
        )

        self.user_prefix = "[user]: "
        self.assistant_prefix = f"[assistant]:"

        self.window = tk.Tk()
        self.window.title("LLM Chatbot")

        self.create_widgets()

    def create_widgets(self):
        self.question_label = tk.Label(self.window, text="User Prompt:")
        self.question_label.pack()

        self.question_entry = tk.Entry(self.window, width=100)
        self.question_entry.pack()

        self.answer_label = tk.Label(self.window, text="Assistant Response:")
        self.answer_label.pack()

        self.answer_text = tk.Text(self.window, height=10, width=100, wrap="word")
        self.answer_text.pack()

        self.submit_button = tk.Button(self.window, text="Submit", command=self.submit_question)
        self.submit_button.pack()

        self.history_label = tk.Label(self.window, text="Question History:")
        self.history_label.pack()

        self.history_text = tk.Text(self.window, height=30, width=100, wrap="word")
        self.history_text.pack()

    def submit_question(self):
        question = self.question_entry.get().strip()
        # Clear the entry fields
        self.question_entry.delete(0, tk.END)
        # Run the question-answering in a separate thread to prevent GUI freezing
        Thread(target=partial(self.get_answer, question)).start()

    def get_answer(self, question):
        # Clear CPU memory before each question
        gc.collect()

        generator = self.generate(self.llm, self.generation_config, question)

        answer = ""
        for word in generator:
            answer += word

        complete_answer = f"Question:\n{question}\n\nAnswer:\n{answer}"
        self.display_answer(complete_answer)
        self.update_history(question,complete_answer)

    def generate(self, llm, generation_config, user_prompt):
        """Run model inference, will return a Generator if streaming is true"""
        return llm(
            self.format_prompt(user_prompt),
            **asdict(generation_config),
        )

    def format_prompt(self, user_prompt):
        return f"""### Instruction:
{user_prompt}

### Response:"""

    def display_answer(self, answer):
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, answer)
        self.answer_text.tag_config("answer_tag", foreground="blue")
        self.answer_text.tag_add("answer_tag", "1.0", tk.END)

    def update_history(self, question, answer):
        self.question_history.append(question)
        self.answer_history.append(answer)

        self.history_text.delete(1.0, tk.END)
        for i in range(len(self.question_history)):
            q = self.question_history[i]
            a = self.answer_history[i]
            self.history_text.insert(tk.END, f"Question {i + 1}: ", "question_tag")
            self.history_text.insert(tk.END, f"{a}\n", "answer_tag")
            self.history_text.insert(tk.END, "\n")

        self.history_text.tag_config("question_tag", foreground="red")
        self.history_text.tag_config("answer_tag", foreground="blue")

    def run(self):
        # Start the GUI main loop
        self.window.mainloop()


if __name__ == "__main__":
    chatbot_gui = ChatbotGUI()
    chatbot_gui.run()

       
