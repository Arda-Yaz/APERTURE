from llm import chat

messages = [
    {
        "role": "system",
        "content": "You are APERTURE, a local personal AI assistant."
    }
]

while True:
    user_input = input("You > ")

    if user_input.lower() in {"exit", "quit"}:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    answer = chat(messages)

    messages.append({
        "role": "assistant",
        "content": answer
    })

    print(f"\nAPERTURE > {answer}\n")