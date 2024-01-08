import re
import urllib
import warnings
from pathlib import Path

import backoff
import pandas as pd
import PyPDF2
import ratelimit
from google.api_core import exceptions
from tqdm import tqdm
from vertexai.language_models import TextGenerationModel

warnings.filterwarnings("ignore")

@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-pro")
    return text_model_pro

def get_gemini_pro_text_response( model: GenerativeModel,
                                  contents,
                                  generation_config: GenerationConfig,
                                  stream=True):
    
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    
    responses = model.generate_content(contents,
                                       generation_config = generation_config,
                                       safety_settings=safety_settings,
                                       stream=True)

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

def model_with_limit_and_backoff(**kwargs):
        return generation_model.predict(**kwargs)

def reduce(initial_summary, prompt_template):
    # Concatenate the summaries from the inital step
    concat_summary = "\n".join(initial_summary)

    # Create a prompt for the model using the concatenated text and a prompt template
    prompt = prompt_template.format(text=concat_summary)

    # Generate a summary using the model and the prompt
    summary = model_with_limit_and_backoff(prompt=prompt, max_output_tokens=1024).text

    return summary

st.header("Vertex AI Gemini API", divider="rainbow")
text_model_pro = load_models()

tab1= st.tabs(["Sumarizar Arquivos"])



with tab1:
    st.write("Using LLM - Text only model")
    st.subheader("Sumarização de Arquivos")
    
    # Story premise
    arquivo = st.file_uploader("Coloque aqui o arquivo",key="arquivo")

    initial_prompt_template = """
    Write a concise summary of the following text delimited by triple backquotes.

    ```{text}```

    CONCISE SUMMARY:
        """

    final_prompt_template = """
            Write a concise summary of the following text delimited by triple backquotes.
            Return your response in bullet points which covers the key points of the text.

            ```{text}```

            BULLET POINT SUMMARY:
        """

    generate_t2t = st.button("Corrija a redação", key="generate_t2t")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Corrigindo..."):
            first_tab1 = st.tabs(["Correção"])
            with first_tab1: 
                # response = get_gemini_pro_text_response(
                #     text_model_pro,
                #     contents,
                #     generation_config=generation_config,
                # )
                reader = PyPDF2.PdfReader(arquivo)
                pages = reader.pages

                # Create an empty list to store the summaries
                initial_summary = []

                # Iterate over the pages and generate a summary for each page
                for page in tqdm(pages):
                    # Extract the text from the page and remove any leading or trailing whitespace
                    text = page.extract_text().strip()

                    # Create a prompt for the model using the extracted text and a prompt template
                    prompt = initial_prompt_template.format(text=text)

                    # Generate a summary using the model and the prompt
                    summary = model_with_limit_and_backoff(prompt=prompt, max_output_tokens=1024).text

                    # Append the summary to the list of summaries
                    initial_summary.append(summary)

                response = reduce(initial_summary, final_prompt_template)

                if response:
                    st.write("Sua redação corrigida:")
                    st.write(response)
                
