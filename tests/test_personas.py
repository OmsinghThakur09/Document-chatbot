from src import chatbot


PERSONAS = {
    "pirate": "You are a pirate. Answer everything in pirate speak.",
    "teacher": "You are a patient teacher explaining to a 10-year-old.",
    "cto": "You are a senior CTO giving terse, opinionated technical advice.",
    "angry dad": "You are a frustrated father explaining to your son"
}


def persona_exp(prompt: str):
    """
    now system will be change and question will remain same and see how LLM model gives
    different answers.
    :param prompt: same user question asked in ask_gpt function
    :return: answer in different personas system.
    """
    for name, system in PERSONAS.items():
        print(f"\n{'─' * 40}")
        print(f"  PERSONA: {name.upper()}")
        print(f"{'─' * 40}")
        print(chatbot.ask_gpt(prompt, system=system))


if __name__ == "__main__":
    persona_exp("what is Gemini?")
