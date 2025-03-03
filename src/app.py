from typing import Dict, List

import gradio as gr
from langgraph_sdk import get_sync_client

langgraph_client = get_sync_client(url="http://localhost:2024/")

assistants = langgraph_client.assistants.search()


def respond(message: str, history: List, run_config: Dict):
    if not run_config.get("thread"):
        run_config["thread"] = langgraph_client.threads.create()

    stream = langgraph_client.runs.stream(
        assistant_id=run_config["assistant"]["assistant_id"],
        thread_id=run_config["thread"]["thread_id"],
        input={"messages": message},
        stream_mode="messages",
    )
    for chunk in stream:
        if chunk.event == "messages/partial":
            yield chunk.data[0]["content"], run_config


with gr.Blocks() as demo:
    with gr.Tab("RAG Agent"):
        state = gr.State({"assistant": assistants[0]})
        chat = gr.ChatInterface(
            fn=respond,
            type="messages",
            additional_inputs=[state],
            additional_outputs=[state],
        )


if __name__ == "__main__":
    demo.launch()
