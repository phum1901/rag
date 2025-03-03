# Doc grader instructions
doc_grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

doc_grader_user_prompt = (
    "Retrieved document: \n\n {document} \n\n User question: {query}"
)

query_rewrite_system_prompt = """You a question re-writer that converts an input question to a better version that is optimized \n
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

query_rewrite_user_prompt = (
    """Here is the initial question: \n\n {query} n Formulate an improved question."""
)

retrieve_system_prompt = """"""
