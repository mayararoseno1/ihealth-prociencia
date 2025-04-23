import json
import os
import pandas as pd

# Caminho do arquivo CSV dentro da pasta "data"
csv_file = "data\Leukemia_GSE9476.csv"  # Substitua pelo nome correto do arquivo
json_file = "data/Leukemia.json"  # Também pode salvar o JSON na mesma pasta

# Verifica se o arquivo CSV existe
if not os.path.exists(csv_file):
    print("Arquivo CSV não encontrado.")
else:
    # Lê o arquivo CSV
    df = pd.read_csv(csv_file)

    # Converte o DataFrame em uma lista de dicionários
    data = df.to_dict(orient="records")

    # Salva os dados no formato JSON
    with open(json_file, mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Arquivo JSON salvo em {json_file}")
