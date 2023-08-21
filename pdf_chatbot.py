import gradio as gr
import random
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai

def upload_file(files):
    global knowledge_base
    file_paths = [file.name for file in files]
    pdf_reader = PdfReader(file_paths[0])
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings(open_ai_key = 'chatgpt api key')
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return file_paths

def respond(message, chat_history):
    user_question = message
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
    docs = docs[:-1]
    llm = OpenAI(open_ai_key = 'chatgpt api key')
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)

    time.sleep(2)
    bot_message = response
    chat_history.append((message, bot_message))
    return "", chat_history

with gr.Blocks(theme='finlaymacklon/smooth_slate') as demo:
    title =gr.HTML("<h1 style='text-align:center; color: #c8c7d0;font-weight:700'> 국민은행 Co-worker AI</h1>")
    description =gr.HTML("""<div style='text-align:center'><div style='display:inline-block;vertical-align:top'>
<a href='https://imgbb.com/'><img src='https://i.ibb.co/WWVQ3SN/kb-logo.webp' alt='kb-logo' style='width:95px; height:95px;margin-right:25px;border-radius:11px;'></a>
</div>
<div style="display:inline-block;margin-top: 14px;">
<p style='color: #ffbc00;font-family: -apple-system, BlinkMacSystemFont, sans-serif;font-weight:750;font-size:40px'>KB COPT</p>
</div></div>""")
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf", "text"], file_count="multiple")
    knowledge_base = upload_button.upload(upload_file, upload_button, file_output)

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, file_output, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()