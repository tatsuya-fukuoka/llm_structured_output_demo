# llm_structured_output_demo

LLM の構造化出力（`response_format` の `json_schema` ＋ Pydantic）が、**形式だけでなく中身（判断の妥当性）まで保証するのか**を、Google Colab 上で検証するデモです。Ollama と vLLM の両方で、同じ実験を再現できます。

題材は「カスタマーサポート問い合わせのトリアージ」です。問い合わせ文から、カテゴリ・緊急度・感情・要約・対応案・エスカレーション要否を JSON で返させ、10 件の正解つきデータで評価します。

## 分かったこと

- スキーマ強制ありの全条件で、**パースは 20/20 成功**した（形式は守られる）
- ただし、スキーマだけを強制した条件では、`urgency`（緊急度）が**ほぼ常に 1 になった**
- 原因は、`response_format` のスキーマが出力を縛る文法にだけ使われ、**プロンプトには入っていない**ためと推定される。`Field(description=...)` に書いた基準も、モデルには届かない
- **スキーマ（または基準）をプロンプトにも入れる**と、緊急度が範囲内に収まる件数が大きく改善した
- この傾向は、Ollama と vLLM の両方で再現した

## ノートブック

| ファイル | バックエンド | モデル | 推奨ランタイム |
| --- | --- | --- | --- |
| `ollama_structured_output_demo_v3.ipynb` | Ollama | `qwen3.5:9b` | T4 GPU |
| `vllm_structured_output_demo.ipynb` | vLLM | `Qwen/Qwen3.5-9B`（24GB 級 GPU）／ `Qwen/Qwen3.5-4B`（それ未満の GPU） | L4 / A100 推奨（T4 でも 4B で動作） |

Colab で開く（ブランチが `main` の場合）:

- [Ollama 版](https://colab.research.google.com/github/tatsuya-fukuoka/llm_structured_output_demo/blob/main/ollama_structured_output_demo_v3.ipynb)
- [vLLM 版](https://colab.research.google.com/github/tatsuya-fukuoka/llm_structured_output_demo/blob/main/vllm_structured_output_demo.ipynb)

## 実験の内容

### 出力スキーマ

| クラス | 内容 |
| --- | --- |
| `TicketTriageBasic` | 説明なしの簡易スキーマ |
| `TicketTriageResult` | `reasoning`（理由）を先頭に置き、`urgency` の基準などを `Field(description=...)` に書いたスキーマ |

### 比較する 5 条件

| 条件 | スキーマ強制 | プロンプトにスキーマ | 基準・reasoning |
| --- | --- | --- | --- |
| A. プロンプトのみ | なし | あり | あり |
| B. スキーマ強制（簡易） | あり | なし | なし |
| B2. B ＋プロンプトにもスキーマ | あり | あり | なし |
| C. スキーマ強制＋基準＋reasoning | あり | なし | あり |
| D. C ＋プロンプトにもスキーマ | あり | あり | あり |

### 評価

10 件の問い合わせ（各 2 回、合計 20 件）を、次の指標で評価します。

- パース成功
- カテゴリ一致
- 緊急度が許容範囲内か
- エスカレーション一致

## 結果

各セルは「一致した件数 / 評価した件数」です（エスカレーションの分母は 14）。

**Ollama（`qwen3.5:9b`）**

| 条件 | カテゴリ一致 | 緊急度が範囲内 | エスカレ一致 |
| --- | --- | --- | --- |
| A | 20/20 | 18/20 | 12/14 |
| B | 20/20 | 10/20 | 12/14 |
| B2 | 20/20 | 16/20 | 14/14 |
| C | 18/20 | 8/20 | 10/14 |
| D | 20/20 | **18/20** | 12/14 |

**vLLM（`Qwen/Qwen3.5-4B`、float16、T4）**

| 条件 | カテゴリ一致 | 緊急度が範囲内 | エスカレ一致 |
| --- | --- | --- | --- |
| A | 18/20 | 18/20 | 10/14 |
| B | 18/20 | 12/20 | 14/14 |
| B2 | 20/20 | 14/20 | 14/14 |
| C | 18/20 | 8/20 | 14/14 |
| D | 18/20 | **18/20** | 10/14 |

パース成功は、どちらも全条件で 20/20 でした。

## 使い方

1. 上のリンクから、使いたい方のノートブックを Colab で開きます。
2. ランタイムのタイプを GPU に変更します（vLLM で 9B を使う場合は L4 または A100）。
3. 上のセルから順に実行します。最後に、5 条件の比較表と、外れたケースが表示されます。

実行に必要な環境は、すべてノートブックの最初のセルでインストールします（Ollama 版は Ollama 本体、vLLM 版は `vllm`、共通で `langchain-openai` と `pydantic`）。

### 実行時の注意

- **Ollama 版**: thinking の無効化に `reasoning_effort="none"` を使います。効かない場合は、`ollama -v` でバージョンを確認してください。
- **vLLM 版**: thinking の無効化に `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` を使います。
- **vLLM 版を T4（16GB）で動かす場合**: Qwen3.5-9B は載らないため、自動で 4B（float16）に切り替わります。CUDA グラフ取得時のメモリ不足を避けるため、`--enforce-eager` を付けて起動します。起動には時間がかかることがあります（最大 20 分待つ設定です）。
- 起動に失敗した場合は、ノートブックがログの末尾を表示します。詳細は `/content/vllm.log` を確認してください。

## 注意事項

- **評価の規模が小さい**: `seed` 固定と `temperature=0` のため、同じ問い合わせの 2 回分はほぼ同じ結果になり、実質のサンプル数は約 10 件です。差が数件なら、誤差の範囲です。
- **正解ラベルは作者が付けたもの**です。
- **Ollama 版と vLLM 版でモデルが違います**（9B の量子化と 4B の float16）。バックエンドの差だけを比べたものではありません。
- 「スキーマがプロンプトに入らない」ことは、条件を変えた結果からの推定で、実装のコードで確かめたものではありません。
- 1 つのタスクの結果であり、他のタスクで同じとは限りません。

## 最小の対策

スキーマ強制は、形式の保証として使います。判断の精度を保つには、システムプロンプトにもスキーマ（または基準）を入れます。

```python
system += "\n出力は次のJSON Schemaに従うJSONのみ。\n" + json.dumps(
    schema.model_json_schema(), ensure_ascii=False)
```
