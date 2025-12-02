# Portfólio de Projetos

Portfólio estático com experimentos rápidos de ciência de dados. Inclui uma demo de categorização automática de produtos usando zero-shot classification (NLI) da Hugging Face.

## Como rodar localmente
1. Clone ou baixe este repositório.
2. Sirva os arquivos estáticos (para permitir leitura do JSON) com um servidor simples, por exemplo:
   ```bash
   python -m http.server 8000
   ```
3. Abra `http://localhost:8000/index.html` no navegador para ver o portfólio. A página do projeto fica em `projeto-zero-shot.html`.

## Demo zero-shot
A demo é 100% estática e usa um arquivo pré-gerado (`public/demo_results.json`). Para atualizar ou gerar novos resultados:

```bash
pip install transformers torch pandas
python demo_zero_shot.py
```

O script:
- Lê exemplos de `pedidos.csv` e `produtos.csv` (quando disponíveis) ou usa uma lista interna.
- Aplica um modelo zero-shot NLI (`facebook/bart-large-mnli` por padrão) com rótulos editáveis.
- Expande abreviações para gerar uma descrição legível.
- Salva as previsões no arquivo `public/demo_results.json` consumido pela UI.

Após gerar o JSON, recarregue `projeto-zero-shot.html` no navegador e use o botão **Classificar**.
