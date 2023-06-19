# Repository-finder
This project is a Repository Finder system that returns the most technically complex and challenging repository from that user's profile when given a GitHub user's URL. The Frontend is made using Streamlit package of python
First step includes fetching of the repositories using Github api and PyGithub package of python. Then from the files of a repo content is fetched. After preprocessing the text it is divided into chunks using LangChain.
Finally the chunks are passed to the GPT for evaluation along with a custom prompt and then results are combined for all the chunks

## Installation
Install the necessary libraries specified below using pip

```bash
  pip install langchain==0.0.150 pandas numpy tiktoken textract transformers openai
```
Run the script app.py using this command
```bash
  streamlit run app.py
```
## Demo

![1](https://github.com/omdwid/Repository-finder/assets/94010815/fff7a73c-1e9f-4195-9cef-641696249cf5)

![2](https://github.com/omdwid/Repository-finder/assets/94010815/5c42f423-8eca-4d46-a1c9-e0b12f19fac7)
