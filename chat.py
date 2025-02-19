from functools import partial
from typing import List, Literal

import gradio as gr
from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine  # noqa
from llama_index.core.chat_engine import SimpleChatEngine  # noqa
from llama_index.core.chat_engine import (
    ContextChatEngine,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


def messages_to_prompt(messages):
    messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt.removeprefix("<|begin_of_text|>")


llm = LlamaCPP(
    model_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    temperature=0.0,
    max_new_tokens=512,
    context_window=4096,
    model_kwargs={"n_gpu_layers": -1},
    verbose=False,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=None,
)


def build_index(
    input_files: List[str],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    device: Literal["cpu", "cuda"],
    index: VectorStoreIndex = None,
    verbose: bool = False,
):
    embed_model = HuggingFaceEmbedding(
        model_name=model_name, device=device, cache_folder="/tmp"
    )

    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    loader = SimpleDirectoryReader(
        input_files=input_files,
    )

    documents = loader.load_data(show_progress=verbose)

    nodes = sentence_splitter(documents)

    if index is None:
        index = VectorStoreIndex(nodes, show_progress=True, embed_model=embed_model)
    else:
        index.insert_nodes(nodes)
    print(f"Index count: {len(index.docstore.docs)}")
    return index


def build_index_wrapper(input_files: List[str], state: gr.State, *args, **kwargs):
    state["index"] = build_index(input_files, index=state["index"], *args, **kwargs)

    return input_files, state


def get_chat_engine(index: VectorStoreIndex, llm: LlamaCPP, history: List):
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
    )

    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.0)]

    qa_prompt_tmpl_str = """
    Context: {context_str}
    Question: {query_str}
    Answer:
    """

    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    system_prompt = "You are an assistant for question-answering tasks. Use only the following pieces of retrieved context to answer the question but not prior knowledge. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
    chat_history = (
        [
            ChatMessage(role=message["role"], content=message["content"])
            for message in history
        ]
        if history
        else None
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=vector_retriever,
        system_prompt=system_prompt,
        node_postprocessors=node_postprocessors,
        chat_history=chat_history,
        context_template=qa_prompt_tmpl,
        llm=llm,
    )

    return chat_engine


def stream_chat(message: str, history: List, state: gr.State):
    chat_engine = get_chat_engine(state["index"], llm, history=history)
    answer = ""
    streaming_response = chat_engine.stream_chat(message)
    for token in streaming_response.response_gen:
        answer += token
        yield answer


with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with PDF Upload")

    state = gr.State(value={"index": None})
    with gr.Row():
        with gr.Column(scale=0.25):
            file_paths = gr.Files(interactive=True)
            button = gr.Button("Build Index")

        chat_interface = gr.ChatInterface(
            fn=stream_chat,
            type="messages",
            show_progress="full",
            additional_inputs=[state],
        )

        button.click(
            fn=partial(
                build_index_wrapper,
                chunk_size=512,
                chunk_overlap=32,
                model_name="BAAI/bge-large-en-v1.5",
                device="cuda",
                verbose=True,
            ),
            inputs=[file_paths, state],
            outputs=[file_paths, state],
            show_progress="full",
        )

if __name__ == "__main__":
    demo.launch()
