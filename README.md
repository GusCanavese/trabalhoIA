# Portfólio de Projetos

Portfólio estático com experimentos rápidos de ciência de dados. Inclui uma demo de categorização automática de produtos usando
zero-shot classification (NLI) da Hugging Face.

## O que o programa faz
- Simula um fluxo de **padronização de descrições de produtos**: lê textos sujos, expande abreviações e sugere categorias.
- Usa **zero-shot classification** (modelo NLI da Hugging Face) para atribuir rótulos sem precisar de dataset anotado.
- Gera um arquivo estático (`public/demo_results.json`) com os resultados; a página web apenas **consulta** esse JSON.
- A descrição exibida na demo é criada pelo script `gerar_demo_zero_shot.py` a partir dos dados (ou exemplos internos),
  e não é inventada na hora pelo navegador.
- Quando os arquivos `pedidos.csv` e `produtos.csv` estão presentes, eles são usados como fonte de dados; caso contrário,
  a lista interna de exemplos é usada para gerar o JSON.

## Como rodar localmente
1. Clone ou baixe este repositório.
2. Sirva os arquivos estáticos (para permitir leitura do JSON) com um servidor simples, por exemplo:
   ```bash
   python -m http.server 8000
   ```
3. Abra `http://localhost:8000/apresentacao-zero-shot.html` no navegador para ver o projeto.

## Demo zero-shot
### Como funciona
- A página `apresentacao-zero-shot.html` **não consulta um banco nem roda modelo no navegador**; ela só lê o arquivo
  `public/demo_results.json` gerado previamente.
- O script `gerar_demo_zero_shot.py` procura correspondências nos CSVs disponíveis, expande abreviações para criar uma
  descrição legível e roda o modelo zero-shot para sugerir categorias.
- Ao abrir a demo e clicar em "Classificar", o texto digitado é normalizado e comparado com os exemplos já salvos
  no JSON. O resultado exibido é a correspondência mais próxima e seus rótulos previstos.

### Como atualizar o JSON
Para gerar ou atualizar os resultados:

```bash
pip install transformers torch pandas
python gerar_demo_zero_shot.py
```

Após gerar o JSON, recarregue `apresentacao-zero-shot.html` no navegador e use o botão **Classificar**.

> Dica: se aparecer `NameError: name 'torch' is not defined` ao rodar o script, instale o PyTorch com `pip install torch`. O
> código importa a biblioteca explicitamente para evitar inconsistências no pipeline do Hugging Face.
