import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

st.set_page_config(
    page_title="G2 - Vendas em E-commerce no Brasil",
    page_icon="🛒",
    layout="wide"
)

RAW_URL = "https://raw.githubusercontent.com/AlexandreLouzada/Dados-Simulados-G2/main/datasets_g2_30_temas/simulacao_ecommerce_brasil.csv"

LOCAL_PATH = Path("dados/simulacao_ecommerce_brasil.csv")
DB_PATH = Path("/tmp/ecommerce.db")

def normalizar_coluna(nome: str) -> str:
    """
    Padroniza o nome das colunas:
    - remove acentos;
    - coloca em minúsculo;
    - troca espaços e símbolos por underline.
    """
    nome = str(nome).strip().lower()
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("utf-8")

    for ch in [" ", "-", "/", ".", "(", ")", "%"]:
        nome = nome.replace(ch, "_")

    while "__" in nome:
        nome = nome.replace("__", "_")

    return nome.strip("_")


def formatar_numero(valor):
    """
    Formata números no padrão brasileiro.
    """
    if pd.isna(valor):
        return "-"

    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def encontrar_coluna(df, palavras):
    """
    Busca automaticamente uma coluna pelo nome.
    """
    for col in df.columns:
        if any(palavra in col for palavra in palavras):
            return col
    return None


@st.cache_data
def carregar_dados():
    """
    Carrega a base de dados.
    Primeiro tenta ler da pasta local.
    Se não existir, baixa direto do GitHub.
    """
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LOCAL_PATH.exists():
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_csv(RAW_URL)
        df.to_csv(LOCAL_PATH, index=False)

    df.columns = [normalizar_coluna(c) for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object" or is_string_dtype(df[col]):
            tentativa = (
                df[col]
                .astype(str)
                .str.replace("R$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )

            convertido = pd.to_numeric(tentativa, errors="coerce")

            if convertido.notna().mean() > 0.7:
                df[col] = convertido

    col_ano = encontrar_coluna(df, ["ano", "year"])
    col_mes = encontrar_coluna(df, ["mes", "month"])

    if col_ano and col_mes:
        df["data"] = pd.to_datetime(
            df[col_ano].astype(int).astype(str)
            + "-"
            + df[col_mes].astype(int).astype(str)
            + "-01",
            errors="coerce"
        )

    return df


def salvar_sqlite(df: pd.DataFrame):

    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df.to_sql("vendas_ecommerce", engine, if_exists="replace", index=False)
        return engine

    except Exception as erro:
        st.warning(
            "A persistência em SQLite não foi executada neste ambiente, "
            "mas o dashboard continuará funcionando normalmente."
        )
        st.caption(f"Detalhe técnico: {erro}")
        return None


def aplicar_filtros(df):

    st.sidebar.header("Filtros")

    filtrado = df.copy()

    colunas_categoricas = [
        c for c in df.columns
        if (df[c].dtype == "object" or is_string_dtype(df[c])) and df[c].nunique() <= 80
    ]

    colunas_data = [
        c for c in df.columns
        if is_datetime64_any_dtype(df[c])
    ]

    for col in colunas_categoricas[:6]:
        opcoes = sorted(df[col].dropna().unique().tolist())

        selecionados = st.sidebar.multiselect(
            col.replace("_", " ").title(),
            opcoes,
            default=[],
            placeholder="Selecione uma ou mais opções"
        )

        if selecionados:
            filtrado = filtrado[filtrado[col].isin(selecionados)]

    if "data" in colunas_data:
        data_min = df["data"].min()
        data_max = df["data"].max()

        if pd.notna(data_min) and pd.notna(data_max):
            col_inicio, col_fim = st.sidebar.columns(2)

            with col_inicio:
                data_inicio = st.date_input(
                    "Data início",
                    value=data_min.date(),
                    min_value=data_min.date(),
                    max_value=data_max.date(),
                    format="DD/MM/YYYY"
                )

            with col_fim:
                data_fim = st.date_input(
                    "Data fim",
                    value=data_max.date(),
                    min_value=data_min.date(),
                    max_value=data_max.date(),
                    format="DD/MM/YYYY"
                )

            if data_inicio > data_fim:
                st.sidebar.error("A data início não pode ser maior que a data fim.")
            else:
                filtrado = filtrado[
                    (filtrado["data"].dt.date >= data_inicio)
                    & (filtrado["data"].dt.date <= data_fim)
                ]

    return filtrado

df = carregar_dados()
salvar_sqlite(df)

st.title("Projeto G2 — Vendas em E-commerce no Brasil")

st.write(
    """
    Este dashboard apresenta uma análise exploratória de dados sobre vendas em e-commerce
    no Brasil. O objetivo é avaliar o comportamento das vendas, pedidos, faturamento,
    ticket médio, categorias, regiões, formas de pagamento e evolução temporal.
    """
)

with st.expander("Visualizar estrutura da base de dados"):
    st.write("Quantidade de linhas e colunas:", df.shape)
    st.write("Colunas identificadas:")
    st.write(list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)

filtrado = aplicar_filtros(df)

numericas = filtrado.select_dtypes(include=np.number).columns.tolist()

categoricas = [
    c for c in filtrado.columns
    if filtrado[c].dtype == "object" or is_string_dtype(filtrado[c])
]

col_faturamento = encontrar_coluna(
    filtrado,
    ["faturamento", "receita", "venda", "valor"]
)

col_pedidos = encontrar_coluna(
    filtrado,
    ["pedido", "pedidos", "compras"]
)

col_ticket = encontrar_coluna(
    filtrado,
    ["ticket"]
)

col_cancelamento = encontrar_coluna(
    filtrado,
    ["cancelamento", "cancelado"]
)

st.subheader("KPIs principais")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if col_faturamento and col_faturamento in numericas:
        st.metric(
            "Faturamento Total",
            f"R$ {formatar_numero(filtrado[col_faturamento].sum())}"
        )
    else:
        st.metric("Total de Registros", len(filtrado))

with col2:
    if col_pedidos and col_pedidos in numericas:
        st.metric(
            "Total de Pedidos",
            formatar_numero(filtrado[col_pedidos].sum())
        )
    else:
        st.metric("Linhas Filtradas", len(filtrado))

with col3:
    if col_ticket and col_ticket in numericas:
        st.metric(
            "Ticket Médio",
            f"R$ {formatar_numero(filtrado[col_ticket].mean())}"
        )
    elif (
        col_faturamento
        and col_pedidos
        and col_faturamento in numericas
        and col_pedidos in numericas
        and filtrado[col_pedidos].sum() != 0
    ):
        ticket_calculado = filtrado[col_faturamento].sum() / filtrado[col_pedidos].sum()
        st.metric(
            "Ticket Médio Calculado",
            f"R$ {formatar_numero(ticket_calculado)}"
        )
    elif numericas:
        st.metric(
            "Média Geral",
            formatar_numero(filtrado[numericas[0]].mean())
        )
    else:
        st.metric("Ticket Médio", "-")

with col4:
    if col_cancelamento and col_cancelamento in numericas:
        st.metric(
            "Taxa Média de Cancelamento",
            f"{formatar_numero(filtrado[col_cancelamento].mean())}%"
        )
    else:
        st.metric("Quantidade de Colunas", filtrado.shape[1])


st.divider()

st.subheader("Análise por categoria ou dimensão")

if numericas and categoricas:
    indice_metrica = 0

    if col_faturamento in numericas:
        indice_metrica = numericas.index(col_faturamento)

    metrica = st.selectbox(
        "Escolha a métrica numérica",
        numericas,
        index=indice_metrica
    )

    dimensao = st.selectbox(
        "Escolha a dimensão de análise",
        categoricas
    )

    resumo = (
        filtrado.groupby(dimensao, as_index=False)[metrica]
        .sum()
        .sort_values(metrica, ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=resumo,
        x=metrica,
        y=dimensao,
        ax=ax
    )

    ax.set_title(
        f"{metrica.replace('_', ' ').title()} por {dimensao.replace('_', ' ').title()}"
    )
    ax.set_xlabel(metrica.replace("_", " ").title())
    ax.set_ylabel(dimensao.replace("_", " ").title())

    st.pyplot(fig)

    st.write(
        """
        Interpretação: os maiores valores representam os grupos com maior participação
        na métrica selecionada. Essa análise permite identificar quais categorias,
        regiões ou segmentos concentram maior desempenho comercial.
        """
    )

    st.dataframe(resumo, use_container_width=True)

else:
    st.warning("Não foram encontradas colunas suficientes para análise categórica.")


st.divider()

st.subheader("Análise temporal")

if "data" in filtrado.columns and numericas:
    metrica_tempo = st.selectbox(
        "Escolha a métrica para evolução temporal",
        numericas,
        key="metrica_tempo"
    )

    serie = (
        filtrado.groupby("data", as_index=False)[metrica_tempo]
        .sum()
        .sort_values("data")
    )

    fig2, ax2 = plt.subplots(figsize=(11, 4))

    sns.lineplot(
        data=serie,
        x="data",
        y=metrica_tempo,
        marker="o",
        ax=ax2
    )

    ax2.set_title(
        f"Evolução temporal de {metrica_tempo.replace('_', ' ').title()}"
    )
    ax2.set_xlabel("Data")
    ax2.set_ylabel(metrica_tempo.replace("_", " ").title())

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    if len(serie) >= 2 and serie[metrica_tempo].iloc[0] != 0:
        variacao = ((serie[metrica_tempo].iloc[-1] / serie[metrica_tempo].iloc[0]) - 1) * 100

        st.write(
            f"""
            Interpretação: no período filtrado, a variação acumulada da métrica
            selecionada foi de {formatar_numero(variacao)}%.
            """
        )
    else:
        st.write(
            "Interpretação: a série temporal permite acompanhar a evolução da métrica ao longo do tempo."
        )

else:
    st.info("A análise temporal depende da existência de colunas de ano e mês na base.")


st.divider()

st.subheader("Correlação estatística")

if len(numericas) >= 2:
    corr = filtrado[numericas].corr(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        ax=ax3
    )

    ax3.set_title("Correlação entre variáveis numéricas")

    st.pyplot(fig3)

    st.write(
        """
        Interpretação: valores próximos de 1 indicam correlação positiva forte,
        valores próximos de -1 indicam correlação negativa forte e valores próximos
        de 0 indicam baixa associação linear entre as variáveis.
        """
    )

else:
    st.info("Não há variáveis numéricas suficientes para gerar matriz de correlação.")


st.divider()

st.subheader("Tabela filtrada")

st.dataframe(filtrado, use_container_width=True)

st.subheader("Conclusão executiva")

st.write(
    """
    A análise permite observar o comportamento das vendas em e-commerce no Brasil
    a partir de diferentes dimensões, como categorias, regiões, formas de pagamento
    e período. Os KPIs auxiliam na leitura rápida do desempenho geral, enquanto os
    gráficos permitem identificar padrões, concentrações e possíveis variações ao
    longo do tempo.

    Como funcionalidade intermediária, o projeto utiliza filtros múltiplos e KPIs
    dinâmicos no Streamlit. Como funcionalidade avançada, utiliza persistência em
    banco SQLite por meio do SQLAlchemy, além de análise de correlação estatística.
    """
)
