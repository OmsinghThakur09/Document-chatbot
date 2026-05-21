import ollama


prompt = "Explain what machine learning is in 3 sentences"

temperatures = [0.0, 0.7, 1.5]
for temp in temperatures:
    print(f"Temperature: {temp}")
    response = ollama.chat(
        model="gemma4:e4b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={'temperature': temp},
    )
    print(response["message"]["content"])
    print("-"*150)
