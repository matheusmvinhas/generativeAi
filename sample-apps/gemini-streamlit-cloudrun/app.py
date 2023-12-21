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

st.header("Vertex AI Gemini API", divider="rainbow")
text_model_pro = load_models()

tab1, tab2 = st.tabs(["Corrigir Redação","Perguntas sobre Notas","Correção de Redação"])

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
    
    prompt = f"""Reescreva a redação apontando os erros de ortografia:
    {redacao}
    """
    prompt2 = f"""
        REDAÇÃO:
        {redacao}

        - Quantas palavras estão escritas erradas na redação?
        - Qual é a nota final subtraindo 5 pontos para cada palavra errada encontrada e sabendo que a nota inicial é 100?
        - Qual é o resultado sabendo que se a nota final for menor que 70 o Resultado é 'REPROVADO 😔' e se for maior ou igual a 70 o Resultado é 'APROVADO 😁'?
        - Reescreva a redação usando a ortografia correta
        Siga o Modelo:
        - Numero de erros: quantidade de palavras erradas
        - Nota final = nota final
        - Resultado = resultado

    """
    generation_config = GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    max_output_tokens=1000,
    )
    contents = [
    prompt
    ]
    contents2 = [
    prompt2
    ]
    generate_t2t = st.button("Corrija a redação", key="generate_t2t")
    generate_t2t2 = st.button("Me de a Nota da Redação", key="generate_grade")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Corrigindo..."):
            first_tab1, first_tab2 = st.tabs(["Correção", "Prompt"])
            with first_tab1: 
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    contents,
                    generation_config=generation_config,
                )
                if response:
                    st.write("Sua redação corrigida:")
                    st.write(response)
            with first_tab2: 
                st.text(prompt)
    if generate_t2t2 and prompt2:
        # st.write(prompt)
        with st.spinner("Gerando nota..."):
            first_tab1, first_tab2 = st.tabs(["Nota", "prompt2"])
            with first_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    contents2,
                    generation_config=generation_config,
                )
                if response:
                    st.write("Sua redação corrigida:")
                    st.write(response)
            with first_tab2:
                st.text(prompt2)
                
with tab2:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Faça perguntas sobre suas notas e presença")
    
    question = st.text_input("Faça sua pergunta \n\n",key="question",value="Qual é a média da nota de matematica?")
    
    prompt = f"""{question}
    """
    generation_config = GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=1000,
    )
    contents = [
    prompt,
    tabela
    ]
    generate_t2t = st.button("Me Responda", key="generate_answer")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Resposta", "Prompt"])
        with st.spinner("Gerando sua resposta..."):
            with second_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    contents,
                    generation_config=generation_config,
                )
                if response:
                    st.write("Sua resposta:")
                    st.write(response)
            with second_tab2:
                st.text(prompt)

with tab3:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Correção de Redação")

    nome_aluno = st.text_input("Qual é o seu nome? \n\n",key="student_name",value="João")
    redacao = st.text_input("Cole aqui sua redação \n\n",key="redacao",value="Redação")
    prompt = f"""Seguindo os critérios:
        1°- Nota Inicial = 100
        2°- Erro de ortografia = 10
        3°- Nota final = Nota Inicial - (Número de erros de ortografia*10)
        4°- Se a Nota Final for menor que 70 o Resultado é 'REPROVADO 😔' e se for maior ou igual a 70 o Resultado é 'APROVADO 😁'

        E analisando:
        '{redacao}'

        Me responda as seguintes perguntas:
        - Quantos erros foram encontrados ?
        - Quais foram os erros encontrados ?
        - Qual é a nota Final ?
        - Qual é o resultado ?
        Entregue as respostas seguindo o modelo:

        Nome Aluno : {nome_aluno}
        Quantidade de Erros:
        Nota Final:
        Resultado:
    """
    generation_config = GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    max_output_tokens=1000,
    )
    contents = [
    prompt,
    tabela
    ]
    generate_t2t = st.button("Corrigir Redação", key="generate_answer")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Resposta", "Prompt"])
        with st.spinner("Gerando sua resposta..."):
            with second_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    contents,
                    generation_config=generation_config,
                )
                if response:
                    st.write("Sua resposta:")
                    st.write(response)
            with second_tab2:
                st.text(prompt)