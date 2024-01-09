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

query_dist = """
Select Distinct VENDEDOR FROM `prj-p-ucbr-prod-ia-6ae3.demoRAGQaRaiaDrogasil.SalesRaiaDrogasilOBT`
"""
df_dist = client.query(query_dist).to_dataframe()
vendedores_lista = df_dist['VENDEDOR'].tolist()

st.write("Using Gemini Pro - Text only model")
st.subheader("Faça perguntas sobre as vendas")

question = st.text_input("Faça sua pergunta \n\n",key="question",value="Qual é a média da nota de matematica?")

prompt = f"""CONTEXTO: Com base na tabela de dados com as colunas :
ID_VENDA STRING Primary Key for the sale record.
DATA DATE Date of the sale.
SKU	STRING Stock Keeping Unit of the product.
PRODUTO	STRING Name of the product.
VENDEDOR STRING Name of the salesperson.
LOJA STRING Store where the sale occurred.
REGIAO STRING Region of the store.
ESTADO STRING State of the store.
CIDADE STRING City of the store.
PRECO NUMERIC Original price of the product.
DESCONTO STRING Discount applied to the product (if any).
PRECO_VENDA NUMERIC Final price after discount.

Responda a pergunta:
{question}
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
    vendedor_es = st.selectbox('Qual Vendedor ?', vendedores_lista)
    if vendedor_es:
        query = f"""
        SELECT * FROM `prj-p-ucbr-prod-ia-6ae3.demoRAGQaRaiaDrogasil.SalesRaiaDrogasilOBT` where VENDEDOR = '{vendedor_es}'
        """
        df = client.query(query).to_dataframe()
        string_representation = df.to_string()
        tabela = string_representation
    else:
        query = f"""
        SELECT * FROM `prj-p-ucbr-prod-ia-6ae3.demoRAGQaRaiaDrogasil.SalesRaiaDrogasilOBT`
        """
        df = client.query(query).to_dataframe()
        string_representation = df.to_string()
        tabela = string_representation

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
