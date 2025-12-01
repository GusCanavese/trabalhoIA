import pandas as pd
import csv, re
from collections import defaultdict, Counter



# !parte do tratamento de dados para leitura do código! #
# Lê CSV tentando diferentes separadores
def ler_csv_automatico(caminho: str) -> pd.DataFrame:
    for sep in [",", ";", "|", "\t"]:
        try:
            return pd.read_csv(caminho, sep=sep, engine="python")
        except Exception:
            pass
    return pd.read_csv(caminho, engine="python")

# Extrai descrições de produtos do campo 'itens'
def extrair_produtos(texto):
    if not isinstance(texto, str):
        return []
    nomes = re.findall(r'descric[aã]o["\']?\s*[:=]\s*["\']([^"\']+)["\']', texto, flags=re.IGNORECASE)
    if not nomes:
        nomes = re.findall(r'descricao"?\s*[:=]\s*"([^"]+)"', texto, flags=re.IGNORECASE)
    return [n.strip() for n in nomes if n and n.strip()]


# Carregar dados básicos
df_clientes = ler_csv_automatico("clientes.csv")
df_produtos = ler_csv_automatico("produtos.csv")

# Ler pedidos e reconstruir coluna 'itens'
with open("pedidos.csv", "r", encoding="utf-8", errors="ignore", newline="") as arquivo:
    leitor = csv.reader(arquivo, delimiter=",", quotechar='"', skipinitialspace=True)
    cabecalho = next(leitor)
    indice_itens = cabecalho.index("itens") if "itens" in cabecalho else len(cabecalho) - 1
    colunas_mantidas = cabecalho[:indice_itens] + ["itens"]

    linhas = []
    for linha in leitor:
        if len(linha) <= indice_itens:
            linha = linha + [""] * (indice_itens + 1 - len(linha))
        texto_itens = ",".join(linha[indice_itens:])
        base = linha[:indice_itens]
        linhas.append(base + [texto_itens])

df_pedidos = pd.DataFrame(linhas, columns=colunas_mantidas)

# Normalizar destinatário e extrair lista de produtos
df_pedidos["destinatario_normalizado"] = df_pedidos.get("destinatario", df_pedidos.columns[3]).astype(str).str.upper().str.strip()
df_pedidos["lista_produtos"] = df_pedidos["itens"].apply(extrair_produtos)
df_pedidos = df_pedidos[df_pedidos["lista_produtos"].map(len) > 0].copy()

# Contagens de itens e pares
contagem_pares = defaultdict(int)
contagem_itens = Counter()

for produtos_unicos in df_pedidos["lista_produtos"]:
    unicos = list(dict.fromkeys([p.upper().strip() for p in produtos_unicos]))
    for a in unicos:
        contagem_itens[a] += 1
    for i in range(len(unicos)):
        for j in range(len(unicos)):
            if i == j:
                continue
            contagem_pares[(unicos[i], unicos[j])] += 1

# Calcula ranking por confiança condicional A→B
def pontuar_proximo(carrinho):
    carrinho = [c.upper().strip() for c in carrinho if c]
    candidatos = defaultdict(float)

    for a in carrinho:
        qtd_a = contagem_itens.get(a, 0)
        if qtd_a == 0:
            continue
        for (x, y), c in contagem_pares.items():
            if x == a:
                candidatos[y] += c / qtd_a

    for a in carrinho:
        candidatos.pop(a, None)

    return sorted(candidatos.items(), key=lambda kv: kv[1], reverse=True)


# Recomendar por cliente (usa o último pedido)
recomendacoes = []
for nome_cliente, grupo in df_pedidos.groupby("destinatario_normalizado", sort=False):
    ultimo_carrinho = grupo.iloc[-1]["lista_produtos"]
    ranqueados = pontuar_proximo(ultimo_carrinho)
    topo = ranqueados[0][0] if ranqueados else ""
    pontuacao_topo = ranqueados[0][1] if ranqueados else 0.0
    alternativas = [it for it, _ in ranqueados[:5]]

    recomendacoes.append({
        "cliente": nome_cliente,
        "ultimos_itens": "; ".join(ultimo_carrinho),
        "recomendacao_top1": topo,
        "score": round(pontuacao_topo, 4),
        "alternativas_top5": "; ".join(alternativas)
    })

pd.DataFrame(recomendacoes).to_csv("recomendacoes.csv", index=False)
