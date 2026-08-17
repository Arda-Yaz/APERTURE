from llm import chat

messages = [
{
    "role": "system",
    "content": """
You are APERTURE, an autonomous local AI agent.

Complete the user's actual goal.

Use tools whenever necessary.
You may use multiple tools sequentially.

Do not tell the user how to perform a task when you can perform it yourself.

Tool results are observations, not necessarily the end of the task.

Keep final answers short and natural.
"""
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

    print("\nAPERTURE > ", end="", flush=True)

    answer = chat(messages)

    print(answer)
    print()