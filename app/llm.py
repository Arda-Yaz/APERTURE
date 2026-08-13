from ollama import chat as ollama_chat
from tools import list_directory

MODEL = "qwen3:8b"

TOOLS = [list_directory]


def chat(messages):
    response = ollama_chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        think=False,
    )

    messages.append(response.message)

    # Tool çağrısı varsa
    if response.message.tool_calls:
        for call in response.message.tool_calls:
            if call.function.name == "list_directory":
                path = call.function.arguments["path"]

                print(f"\n[REQUEST] Read directory: {path}")
                permission = input("Allow? (y/n): ")

                if permission.lower() == "y":
                    result = list_directory(path)
                else:
                    result = "Permission denied by user."

                messages.append({
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": result,
                })

        # Tool sonucunu gördükten sonra final cevap
        final_response = ollama_chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,
        )

        messages.append(final_response.message)

        return final_response.message.content

    return response.message.content