"""
Gera resultados de zero-shot classification para a demo estática.
Execute localmente para atualizar public/demo_results.json.
"""

from pathlib import Path
import json
import re
from typing import List

import pandas as pd
from transformers import pipeline

LABELS = [
    "Eletrodomésticos",
    "Utensílios",
    "Peças",
    "Ferramentas",
    "Informática",
    "Limpeza",
    "Acessórios",
]

MODEL_NAME = "facebook/bart-large-mnli"

ABBREVIATIONS = {
    "liq": "liquidificador",
    "liq.": "liquidificador",
    "ind": "industrial",
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
    "lav": "lavadora",
    "asp": "aspersão",
    "cont": "contato",
    "inox": "inox",
    "coz": "cozinha",
}


def load_examples() -> List[str]:
    examples: List[str] = []
    pedidos = Path("pedidos.csv")
    produtos = Path("produtos.csv")

    def read_column(path: Path) -> List[str]:
        if not path.exists():
            return []
        df = pd.read_csv(path)
        for col in [
            "descricao",
            "Descrição",
            "Descricao",
            "item",
            "Item",
            "ITEM",
            "produto",
            "Produto",
        ]:
            if col in df.columns:
                series = df[col].dropna().astype(str).tolist()
                return series
        return []

    examples.extend(read_column(pedidos)[:6])
    examples.extend(read_column(produtos)[:6])

    fallback = [
        "LIQ. IND. 4LTS LQL4",
        "KIT FERRAM. MANUT PNC",
        "HD SSD 1TB NVME",
        "DET.LIQ CLOR 5L",
        "CABO REDE CAT6 15M",
        "SUPR LIMPA CONT. ELETR",
        "BICO ASP LAV IND",
        "FACA INOX COZ 8\"",
        "PLACA CONTRL MOTOR 220V",
        "NOTE I5 16GB 512",
    ]

    if len(examples) < 10:
        examples.extend(fallback)

    unique_examples = []
    seen = set()
    for text in examples:
        if text not in seen:
            seen.add(text)
            unique_examples.append(text)
    return unique_examples[:15]


def humanize_description(text: str) -> str:
    clean = re.sub(r"[^\w\s\"]", " ", text)
    tokens = clean.split()
    expanded = []
    for token in tokens:
        lower = token.lower()
        replacement = ABBREVIATIONS.get(lower)
        expanded.append(replacement or token.capitalize())
    return " ".join(expanded)


def main() -> None:
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    examples = load_examples()
    results = []

    for text in examples:
        prediction = classifier(text, LABELS, multi_label=False)
        labels = prediction["labels"]
        scores = prediction["scores"]
        top = [
            {"label": label, "score": float(score)}
            for label, score in zip(labels[:3], scores[:3])
        ]
        results.append(
            {
                "texto": text,
                "categoria": top[0]["label"],
                "score": float(top[0]["score"]),
                "candidatas": top,
                "descricao_legivel": humanize_description(text),
            }
        )

    output_path = Path("public/demo_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print(f"Gerado {len(results)} exemplos em {output_path}")


if __name__ == "__main__":
    main()
