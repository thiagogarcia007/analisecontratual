import os
import csv
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from groq import Groq

# ================================================
# Carregar variáveis de ambiente (opcional)
# ================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================================================
# Ler CSV de usuários (para autenticação)
# ================================================
users_file = "users.csv"
users_data = []
with open(users_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        users_data.append(row)

def authenticate_user(username: str, password: str) -> bool:
    for user in users_data:
        if user["username"] == username and user["password"] == password:
            return True
    return False

# ================================================
# Ler CSV de contratos (qa_with_id_first_column.csv)
# ================================================
data_file = "qa_with_id_first_column.csv"
rows = []
with open(data_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# ================================================
# Inicializar LLMs ou outra IA
# ================================================
def initialize_embeddings(provider="openai"):
    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        return llm
    elif provider == "groq":
        groq_client = Groq(api_key=GROQ_API_KEY)
        return groq_client
    else:
        raise ValueError("Provedor inválido. Use 'openai' ou 'groq'.")

# ================================================
# Função para localizar o dicionário do CSV via ID
# ================================================
def get_csv_row_by_id(selection: str):
    contract_id = selection.split("-")[0].strip()  # ex.: '5'
    for r in rows:
        if r["id"].strip() == contract_id:
            return r
    return None

# ================================================
# Geração de resposta final
# ================================================
def generate_response(pdf_text: str, selected_contract: str, llm_or_groq, analysis_mode: str) -> str:
    row = get_csv_row_by_id(selected_contract)
    if not row:
        return "Não encontrei esse tipo de contrato no CSV."

    best_practice = (
        f"Tipo de contrato: {row.get('tipo_contrato')}\n"
        f"Objetivo: {row.get('objetivo')}\n"
        f"Requisitos obrigatórios: {row.get('requisitos_obrigatorios')}\n"
        f"Requisitos opcionais: {row.get('requisitos_opcionais')}\n"
    )

    if analysis_mode == "Análise de Requisitos":
        prompt = f"""
        [PROMPT DE ANÁLISE DE REQUISITOS...]
        """
    else:
        prompt = f"""
        [PROMPT DE ANÁLISE COMPLETA...]
        """

    if isinstance(llm_or_groq, ChatOpenAI):
        response = llm_or_groq(prompt)
        return response.content if hasattr(response, "content") else "Erro ao processar."
    elif hasattr(llm_or_groq, "chat"):
        messages = [
            {"role": "system", "content": "Você é um assistente jurídico."},
            {"role": "user", "content": prompt}
        ]
        chat_completion = llm_or_groq.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        if hasattr(chat_completion, "choices") and len(chat_completion.choices) > 0:
            return chat_completion.choices[0].message.content
        else:
            return "Erro ao processar a resposta com a API GROQ."
    else:
        return "Provedor de IA inválido ou não suportado."

# ================================================
# Processar PDF
# ================================================
def process_pdf(file) -> str:
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
        return ""

# ================================================
# MAIN
# ================================================
def main():
    # CSS para clarear o textarea desabilitado
    st.markdown("""
    <style>
    textarea[disabled] {
        background-color: #ffffff !important; 
        color: #000000 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(page_title="Texto Manager - Tahech", page_icon=":bird:")
    st.title("Assistente Jurídico - Tahech Advogados")

    # -- Login --
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.subheader("Login")
        username = st.text_input("Usuário:")
        password = st.text_input("Senha:", type="password")
        if st.button("Entrar"):
            if authenticate_user(username, password):
                st.session_state["logged_in"] = True
                st.success(f"Bem-vindo, {username}!")
                st.write("DEBUG: logged_in =", st.session_state["logged_in"])
            else:
                st.error("Usuário ou senha inválidos.")

    else:
        # =============== TUDO VAI DENTRO DESTE 'ELSE' ===============
        st.header("Ferramenta de Análise de Contratos")

        # Aqui você garante que TUDO será visto apenas se já logado.

        if "user_text" not in st.session_state:
            st.session_state["user_text"] = None

        # 1. Provedor
        provider = st.radio("Escolha o provedor de API:", ("openai", "groq"))

        # 2. Modo de Análise
        analysis_mode = st.radio(
            "Selecione o modo de análise:",
            ("Análise de Requisitos", "Análise Completa")
        )

        # 3. Modo de entrada do contrato
        input_mode = st.radio("Modo de entrada do contrato:", ("Carregar PDF", "Inserir Manualmente"))
        if input_mode == "Carregar PDF":
            uploaded_file = st.file_uploader("Carregue um arquivo PDF", type="pdf")
            if uploaded_file:
                st.write("Processando PDF...")
                text = process_pdf(uploaded_file)
                if text:
                    st.session_state["user_text"] = text
                    st.success("Texto processado!")
        else:
            user_input = st.text_area("Digite o texto do contrato:")
            if user_input:
                st.session_state["user_text"] = user_input
                st.success("Texto processado!")

        # 4. Tipo de Contrato
        contract_types = [
            "1 - Contrato de manutenção de serviço",
            "2 - Contrato social",
            "3 - Contrato de compra e venda",
            "4 - Contratos administrativos",
            "5 - Contrato de Consumo ou prestação de serviços",
            "6 - Contrato de sociedade",
            "7 - Contratos mercantis",
            "9 - Contrato eletrônico",
            "10 - Contrato de trabalho",
            "11 - Contrato - Outros diversos"
        ]
        selected_contract = st.selectbox("Selecione a característica contratual (ID - Nome):", contract_types)

        # 5. Botão de Geração
        if st.button("Gerar Análise"):
            if not st.session_state["user_text"]:
                st.warning("Por favor, carregue ou insira o texto do contrato antes de analisar.")
            else:
                st.write("Gerando análise...")
                llm_or_groq = initialize_embeddings(provider)
                result = generate_response(
                    pdf_text=st.session_state["user_text"],
                    selected_contract=selected_contract,
                    llm_or_groq=llm_or_groq,
                    analysis_mode=analysis_mode
                )
                st.subheader("Resposta Gerada")
                st.text_area("Resultado da Análise", value=result, height=300, disabled=True)

    st.write("DEBUG: logged_in =", st.session_state["logged_in"])

if __name__ == '__main__':
    main()
