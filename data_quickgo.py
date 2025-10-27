import requests
import pandas as pd

def fetch_go_terms_from_uniprot(uniprot_id: str) -> pd.DataFrame:
    """
    使用 QuickGO 的 annotation/search 接口获取某个 UniProt 蛋白的 GO 注释。
    返回包含 goId、goName、aspect（BP/MF/CC）、evidenceCode、reference、assignedBy 等字段的 DataFrame。
    """
    url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    headers = {"Accept": "application/json"}

    # QuickGO API 的 limit 参数不能太大（最大约 100），需要分页获取
    all_results = []
    page = 1
    page_size = 100

    while True:
        params = {
            "geneProductId": uniprot_id,  # 直接使用 UniProt ID，不需要加前缀
            "limit": page_size,
            "page": page
        }
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        results = response_data.get("results", [])

        if not results:
            break

        all_results.extend(results)

        # 检查是否还有更多数据
        number_of_hits = response_data.get("numberOfHits", 0)
        if len(all_results) >= number_of_hits:
            break

        page += 1

    # 只保留常用字段，并去重
    # API 返回的字段名是 goAspect 而不是 aspect
    df = pd.DataFrame([{
        "goId": row.get("goId"),
        "aspect": row.get("goAspect")
    } for row in all_results]).drop_duplicates()
    # 方便阅读：按 aspect -> goId 排序
    order = {"molecular_function": 0, "biological_process": 1, "cellular_component": 2}
    df["aspect_order"] = df["aspect"].map(order).fillna(9)
    df = df.sort_values(["aspect_order", "goId"]).drop(columns=["aspect_order"])
    return df

if __name__ == "__main__":
    df = fetch_go_terms_from_uniprot("P61073")  # CXCR4
    print(df.head)
    print(f"Total GO annotations: {len(df)}")