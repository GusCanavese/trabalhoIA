"""
Gera resultados de zero-shot classification para a demo estática.
Execute localmente para atualizar public/demo_results.json.
"""

from pathlib import Path
import json
import re
from typing import List

import pandas as pd
import torch
from transformers import pipeline

# NOVAS CATEGORIAS BASEADAS NA ANÁLISE DOS ITENS INDUSTRIAIS
LABELS = [
    "Cocção e Fornos",           # Para fogões, fornos, fritadeiras, chapas
    "Refrigeração Comercial",    # Para freezers, bebedouros, caixas térmicas
    "Preparo de Alimentos",      # Para liquidificadores, picadores, extratores, misturelas
    "Ventilação e Climatização", # Para exaustores, ventiladores
    "Mobiliário e Exposição",    # Para estufas de salgados, baleiros, mesas, vitrines
    "Limpeza e Organização",     # Para lixeiras, contentores, estrados
    "Automação e Pesagem",       # Para balanças, seladoras, gavetas
    "Utensílios Diversos",       # Para itens menores não categorizados
]

# Modelo Multilíngue (Recomendado para PT-BR se tiver RAM suficiente, senão mantenha o bart-large-mnli)
# O bart-large-mnli funciona bem, mas precisa de um template em inglês ou tradução implícita.
# Vamos manter o original para garantir compatibilidade com seu ambiente, 
# mas ajustaremos o TEMPLATE.
MODEL_NAME = "facebook/bart-large-mnli" 

# Dicionário expandido com base no produtos.csv
ABBREVIATIONS = {
    "liq": "liquidificador",
    "liq.": "liquidificador",
    "ind": "industrial",
    "ind.": "industrial",
    "lt": "litro",
    "lts": "litros",
    "lts.": "litros",
    "pct": "pacote",
    "kit": "kit",
    "manut": "manutenção",
    "ferram": "ferramentas",
    "det": "detergente",
    "clor": "cloro",
    "pnc": "pneumática",
    "note": "notebook",
    "ssd": "ssd",
    "hd": "hd",
    "ctrl": "controle",
    "contrl": "controle",
    "eletr": "elétrico",
    "elet": "elétrico",
    "lav": "lavadora",
    "asp": "aspersão",
    "cont": "contato",
    "inox": "inox",
    "coz": "cozinha",
    "met": "metal",
    "biv": "bivolt",
    "prof": "profissional",
    "ext": "extrator",
    "fde": "fundo",
    "vid": "vidro",
    "frit": "fritadeira",
    "esb": "extrator suco",
    "c/": "com",
    "s/": "sem",
    "ped": "pedal",
    "bco": "banco",
    "pto": "preto",
    "az": "azul",
    "vm": "vermelho",
    "am": "amarelo",
    "auto": "automático"
}

STOPWORDS = {
    "de", "do", "da", "para", "com", "em", "por", "sem", 
    "uma", "um", "na", "no", "nas", "nos", "e", "ou", 
    "a", "o", "as", "os", "me", "ltda", "epp"
}


def load_examples() -> List[str]:
    examples: List[str] = []
    # Tenta carregar dos CSVs se existirem
    pedidos = Path("pedidos.csv")
    produtos = Path("produtos.csv")

    def read_column(path: Path, col_name: str) -> List[str]:
        if not path.exists():
            return []
        try:
            # Tenta ler com tratamento de erro python engine
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8", sep=None)
            # Normaliza colunas para lower case para achar 'descricao' ou 'nome_do_produto'
            df.columns = [c.lower() for c in df.columns]
            
            # Procura pela coluna correta
            target_col = None
            if col_name in df.columns:
                target_col = col_name
            # Fallback para produtos.csv que tem 'nome_do_produto'
            elif "nome_do_produto" in df.columns:
                target_col = "nome_do_produto"
            # Fallback para pedidos.csv que tem 'itens' (precisaria de extração complexa, 
            # mas vamos tentar pegar descrições simples se houver)
            
            if target_col:
                series = df[target_col].dropna().astype(str).tolist()
                return series
        except Exception as e:
            print(f"Erro ao ler {path}: {e}")
        return []

    # Prioriza produtos.csv que tem descrições limpas
    prods = read_column(produtos, "nome_do_produto")
    if prods:
        examples.extend(prods[:400]) # Aumentei o limite para pegar mais variedade
    
    # Se não tiver produtos, tenta fallback manual
    if len(examples) < 5:
        examples = [
            "LIQUIDIFICADOR INDUSTRIAL LS10 BIVOLT",
            "FOGÃO INDUSTRIAL 04 BOCAS ALTA PRESSÃO",
            "LIXEIRA COM PEDAL REDONDA 40LTS",
            "BALANÇA DIGITAL 33KG",
            "EXAUSTOR 50CM LINHA PESADA",
            "ESTUFA CURVA 8 BANDEJAS",
            "CORTADOR DE FRIOS AUTOMATICO",
            "CAIXA TÉRMICA 100 LITROS",
            "MESA INOX DESMONTÁVEL",
            "EXTRATOR DE SUCO PRO"
        ]

    # Remove duplicatas mantendo a ordem
    unique_examples = []
    seen = set()
    for text in examples:
        # Limpeza básica para evitar duplicatas sujas
        clean_text = text.strip()
        if clean_text and clean_text not in seen:
            seen.add(clean_text)
            unique_examples.append(clean_text)
            
    return unique_examples[:200]


def humanize_description(text: str) -> str:
    # Remove caracteres estranhos mas mantem acentos e numeros
    text = str(text)
    clean = re.sub(r"[^\w\s\d\.,\-]", " ", text)
    tokens = clean.split()
    expanded = []
    for token in tokens:
        lower = token.lower().strip(".,")
        # Substituição direta ou mantém original
        replacement = ABBREVIATIONS.get(lower)
        expanded.append(replacement or token) # Mantém case original se não achar
    return " ".join(expanded)


def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    normalized = text.lower()
    return normalized.strip()


def extract_keywords(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]


def build_questions(top_labels: List[str], keywords: List[str]) -> List[str]:
    focus = ", ".join(keywords[:2]) if keywords else "este item"
    return [
        f"Por que este item foi classificado como {top_labels[0]}?",
        f"O termo '{focus}' foi decisivo para excluir {top_labels[-1]}?",
        f"Existe ambiguidade entre {top_labels[0]} e {top_labels[1]}?"
    ]


def main() -> None:
    print(f"Carregando modelo {MODEL_NAME}...")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    
    examples = load_examples()
    print(f"Processando {len(examples)} produtos...")
    
    results = []

    # O hypothesis_template ajuda o modelo a entender o contexto em Português
    # Mesmo o modelo sendo treinado em NLI inglês, ele entende a relação semântica.
    # Traduzimos a intenção para alinhar com os LABELS.
    template = "Este produto pertence à categoria de {}."

    for text in examples:
        readable = humanize_description(text)
        
        # Executa classificação
        prediction = classifier(
            readable, 
            LABELS, 
            multi_label=False,
            hypothesis_template=template
        )
        
        labels = prediction["labels"]
        scores = prediction["scores"]
        
        top = [
            {"label": label, "score": float(score)}
            for label, score in zip(labels[:3], scores[:3])
        ]
        distribution = [
            {"label": label, "score": float(score)}
            for label, score in zip(labels, scores)
        ]
        
        keywords = extract_keywords(readable)
        questions = build_questions(labels, keywords)
        
        results.append(
            {
                "texto": text,
                "categoria": top[0]["label"],
                "score": float(top[0]["score"]),
                "candidatas": top,
                "descricao_legivel": readable,
                "detalhes": {
                    "texto_normalizado": normalize_text(text),
                    "palavras_chave": keywords,
                    "scores_por_categoria": distribution,
                    "perguntas_sugeridas": questions,
                },
            }
        )

    output_path = Path("public/demo_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print(f"Sucesso! Gerado {len(results)} classificações em {output_path}")


if __name__ == "__main__":
    main()