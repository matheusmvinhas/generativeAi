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

tab1, tab2= st.tabs(["Corrigir Redação","Perguntas sobre Notas"])

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
    redacao = st.text_area("Cole aqui sua redação: \n\n",key="redacao",value="texto",height=3000)
    
    prompt = f"""You will act as a teacher correcting an essay. Assume every student starts with a 10.
    Every time that you find a mistake, count that mistake.

    Everything should be in Brazilian Portuguese, and I want the following structure:

    Correções: 'The corrected version of whatever mistakes you find.'

    Erros: 'The number os mistakes made' \n
    Nota Final: '10 minus the number os mistakes made' \n
    Resultado: 'If Nota Final is equal or greater than 7, write "Aprovado 😃", is it is less than 7, write "Reprovado 😞" ' \n

    Also, output the correct version of the essay.

    The essay below should be corrected:

    ---
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
    temperature=0.2,
    top_p=0.93,
    top_k=27,
    candidate_count=1,
    max_output_tokens=2048,
    )
    contents = [
    prompt
    ]
    contents2 = [
    prompt2
    ]
    generate_t2t = st.button("Corrija a redação", key="generate_t2t")
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
                st.text("""
                    Como Funciona:
                    O Gemini irá analisar a redação, econtrar os erros e reescrever com a gramática correta.
                    Composição de nota e resultado:
                        1°- O Aluno começa com nota 10.
                        2°- Cada erro vale 1 ponto
                        3°- Nota final é igual a nota inicial menos a quantidade de erro.
                        4°- Se a nota final for maior ou igual a 7 o resultado é 'Aprovado 😃', Se for menor que 7 o resultado é 'Reprovado 😔'
                    """)
                
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
