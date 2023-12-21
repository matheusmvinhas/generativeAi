import os
import streamlit as st
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image, 
                                                HarmCategory, 
                                                HarmBlockThreshold, 
                                                Part)
import vertexai
from google.cloud import bigquery

PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = bigquery.Client(project=PROJECT_ID)
@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-pro")
    return text_model_pro

def get_gemini_pro_text_response( model: GenerativeModel,
                                  contents: str, 
                                  generation_config: GenerationConfig,
                                  stream=True):
    
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    
    responses = model.generate_content(prompt,
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

st.header("Vertex AI Gemini API", divider="rainbow")
text_model_pro = load_models()

tab1, tab2 = st.tabs(["Corrigir Redação","Perguntas sobre Notas"])

query = """
SELECT * FROM `prj-p-ucbr-prod-ia-6ae3.demoRAGQaSagres.notas_alunos`
"""
df = client.query(query).to_dataframe()
string_representation = df.to_string()
tabela = string_representation

with tab1:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Correção de Redação")
    
    # Story premise
    redacao = st.text_input("Cole aqui sua redação: \n\n",key="redacao",value="texto")
    
    prompt = f""" corrija a seguinte redação, apontanto os erros de forma clara e sugerindo a melhor forma:
    {redacao}
    """
    config = GenerationConfig(
        temperature=temperature,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
    )
    
    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
    
    generate_t2t = st.button("Corrija a redação", key="generate_t2t")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Corrigindo..."):
            first_tab1, first_tab2 = st.tabs(["Correção", "Prompt"])
            with first_tab1: 
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Sua redação corrigida:")
                    st.write(response)
            with first_tab2: 
                st.text(prompt)
                
with tab2:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Faça perguntas sobre suas notas e presença")
    
    question = st.text_input("Faça sua pergunta \n\n",key="question",value="Qual é a média da nota de matematica?")
    
    prompt = f"""{question}
    """
    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
    generate_t2t = st.button("Me Responda", key="generate_answer")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Resposta", "Prompt"])
        with st.spinner("Gerando sua resposta..."):
            with second_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    tabela,
                    generation_config=config,
                )
                if response:
                    st.write("Sua resposta:")
                    st.write(response)
            with second_tab2:
                st.text(prompt)