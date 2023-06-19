from github import Github
import pandas as pd
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain

import ast
import numpy as np
import re
import streamlit as st
from dotenv import load_dotenv, dotenv_values

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
g = Github(os.getenv("GITHUB_KEY")) 


def preprocess(text):
    text = text.replace('\n','')
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub('[^A-Za-z0-9]+',' ',text)
    return text


def get_chunks(text):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Step 4: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    chunks = text_splitter.create_documents([text])
    return chunks

def get_result(chunks):
    query = "You are an AI that can calculate the complexity or technically challenging score given a code.Return a one word numeric answer between 0 to 10 containing the complexity score, larger the value means more complex. Stricly return a one word numeric value"
    results = []
    model = OpenAI(temperature=0.1)

    for chunk in chunks:
        ans = model(f'{query}+{chunk.page_content}')
        results.append(ans)
    
    for i,r in enumerate(results):
        m = re.search(r"\b([0-9]|10)\b", r)
        if m:
            n = m.group(1)
            results[i] = ast.literal_eval(n)
        else:
            results[i] = np.nan
            
    ans = pd.Series(results)
    ans = ans.replace(0,np.nan)
    
    return ans.mean()


# Get all of the contents of the repository recursively
def most_complex_repo(repos):
    maxi = 0
    maxi_repo = ""
    for repo in repos:
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                # Extend the list with the contents of the directory
                contents.extend(repo.get_contents(file_content.path))
            else:
                # Print the decoded content of the file
                try:
                    with open('file_info.txt', 'a') as f:
                        f.write(file_content.decoded_content.decode())
                except:
                    pass
                
        with open('file_info.txt', 'r') as f:
            text = f.read()
        
        # clear the content of the file for the next repo
        f = open('file_info.txt', 'w')
        f.close()
        
        # if the repo is empty then continue
        if len(text) == 0:
            continue
            
        text = preprocess(text)
        chunks = get_chunks(text)
        result = get_result(chunks)
        print(result)
        
        if result > maxi:
            maxi = result
            maxi_repo = repo.name


    return maxi_repo, maxi


st.title("Repository finder")
user_link = st.text_input("Enter GitHub User Link")
user_name = user_link.split('/')[-1]



    
c1, c2, c3 = st.columns(3, gap="medium")

if c2.button('Submit'):
        try:
            user = g.get_user(user_name)
            repos = user.get_repos()
        except:
            st.write("Enter a valid user link")
            
        repo_name, score = most_complex_repo(repos)
        st.text("Most complex repo of the user is: ")
        st.write(f"{user_link}/{repo_name}")
