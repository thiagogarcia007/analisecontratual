import os
import PyPDF2
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

# Carregar variáveis de ambiente
load_dotenv()

# Chaves de API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Inicializar cliente GROQ
groq_client = Groq(api_key=GROQ_API_KEY)

# Carregar o CSV
data_file = "qa_with_id_first_column.csv"
loader = CSVLoader(file_path=data_file)
documents = loader.load()

# ============================================
# Função que localiza a linha de CSV por ID
# ============================================
def get_document_for_contract(selection: str):
    """
    Recebe algo como '5 - Contrato de Consumo ou prestação de serviços'
    Extrai o '5' e localiza a linha do CSV (doc) que inicia com '5,'.
    """
    # Tenta extrair o ID:
    contract_id = selection.split("-")[0].strip()  # '5'

    # Ver o que realmente tem em doc.page_content
    # (Se quiser depurar, pode imprimir doc.page_content)
    for doc in documents:
        splitted = doc.page_content.split(",", 1)  # ['5', 'Contrato de Consumo...,Protege...']
        if splitted[0].strip() == contract_id:
            return doc.page_content  # retorna a linha inteira do CSV
    return None


def initialize_embeddings(provider):
    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(documents, embeddings)
        return db, llm
    elif provider == "groq":
        db = [{"content": doc.page_content} for doc in documents]
        return db, groq_client
    else:
        raise ValueError("Provedor de API inválido. Use 'openai' ou 'groq'.")


def generate_response(pdf_text, contract_type, db, llm, provider):
    # 1) Buscar a linha exata do CSV
    best_practice = get_document_for_contract(contract_type)
    if not best_practice:
        return "Não encontrei esse tipo de contrato no CSV (busca por ID exato)."

    # 2) Construir prompt a partir do PDF + Contrato do CSV
    prompt = f"""
    Você é um assistente virtual especializado em análise de contratos.
    Abaixo, texto do contrato (PDF ou digitado) e requisitos do tipo selecionado.

    TEXTO DO CONTRATO:
    {pdf_text}

    TIPO DE CONTRATO SELECIONADO: {contract_type}

    REQUISITOS (CSV):
    {best_practice}

    Sua tarefa:
      - Identificar PARTES (Contratante, Contratado).
      - Relacionar os requisitos encontrados (citando cláusulas) e os não encontrados.
      - Sugerir melhorias no final.

    Responda no formato:

        PARTES DO CONTRATO
        Contratante: ...
        Contratada:  ...

        REQUISITOS DO CONTRATO
        [REQUISITO 1] [Cláusula X, § Y]
        ...

        Requisitos Não atendidos:
        [REQUISITO N] ...

        [ANÁLISE FINAL DA IA] ...
    """

    # 3) Chamar LLM
    if provider == "openai":
        response = llm(prompt)
        return response.content if hasattr(response, 'content') else "Erro ao processar a resposta."
    elif provider == "groq":
        messages = [
            {"role": "system", "content": "Você é um assistente jurídico e deve se ater ao escopo do contrato."},
            {"role": "user", "content": prompt}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
            return chat_completion.choices[0].message.content
        else:
            return "Erro ao processar a resposta com a API GROQ."
    else:
        return "Provedor de API desconhecido."


def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("Nenhum texto encontrado no PDF.")
        return text
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
        return None


def main():
    st.set_page_config(page_title="Texto Manager - Tahech", page_icon=":bird:")
    st.header("Analisador de Cláusulas Contratuais - Tahech Advogados")

    if "user_text" not in st.session_state:
        st.session_state["user_text"] = None

    provider = st.radio("Escolha o provedor de API:", ("openai", "groq"))
    db, llm = initialize_embeddings(provider)

    input_mode = st.radio("Escolha como deseja fornecer o texto:", ("Carregar PDF", "Inserir Manualmente"))
    if input_mode == "Carregar PDF":
        uploaded_file = st.file_uploader("Carregue um arquivo PDF", type="pdf")
        if uploaded_file:
            st.write("Processando o arquivo PDF...")
            text = process_pdf(uploaded_file)
            if text:
                st.session_state["user_text"] = text
                st.success("Texto processado com sucesso!")
            else:
                st.warning("Não foi possível processar o arquivo.")
    else:
        user_input = st.text_area("Digite o texto do contrato:")
        if user_input:
            st.session_state["user_text"] = user_input
            st.success("Texto processado com sucesso!")

    if st.session_state["user_text"]:
        # Observe como definimos o label do selectbox:
        contract_types = {
            "1 - Contrato de manutenção de serviço": "...",
            "5 - Contrato de Consumo ou prestação de serviços": "...",
            "10 - Contrato de trabalho": "...",
        }
        selected_contract = st.selectbox(
            "Selecione a característica contratual:",
            list(contract_types.keys())
        )

        if st.button("Analisar Informação"):
            st.write("Gerando uma análise com base na seleção do contrato...")
            result = generate_response(
                pdf_text=st.session_state["user_text"],
                contract_type=selected_contract,
                db=db,
                llm=llm,
                provider=provider
            )
            st.markdown("### Resposta Gerada")
            st.write(result)


if __name__ == '__main__':
    main()
