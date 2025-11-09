Excellent — this is *exactly* the stage where AMLSim gets confusing for most people.
You now have all the **raw AMLSim outputs**, but not all of them are directly useful for ML training. Let’s break down exactly what each file means and which ones you actually need 👇

---

## 🗂️ What Each File Is

| File                             | Description                                                                                                                                  | Use for ML Training?                                                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **`accounts.csv`**               | Contains all bank accounts (ID, balance, start/end time, isSAR, etc.). Each row is one account.                                              | ✅ Yes — needed to know which accounts are *suspicious* (SAR) or *normal*.                                                        |
| **`transactions.csv`**           | Full list of transactions between accounts. Each row = one transaction. Contains sender (`orig`), receiver (`dest`), amount, timestamp, etc. | ✅ Yes — this is your **core dataset** for modeling money flow or network structure.                                              |
| **`alert_accounts.csv`**         | Accounts involved in suspicious typologies (part of “alert” subgraphs).                                                                      | ⚙️ Optional — useful if you want to analyze *why* an account was flagged. Can help with *label propagation* or *graph labeling*. |
| **`alert_transactions.csv`**     | Suspicious transactions linked to alerts (i.e., transactions between SAR accounts).                                                          | ✅ Recommended — can serve as *positive samples* for supervised learning.                                                         |
| **`sar_accounts.csv`**           | Subset of `accounts.csv` that are SAR (Suspicious Activity Report) flagged.                                                                  | ✅ Yes — provides *ground-truth suspicious accounts*.                                                                             |
| **`cash_tx.csv`**                | Contains only cash-in / cash-out transactions.                                                                                               | ⚙️ Optional — use only if your model distinguishes cash flow from normal transfers.                                              |
| **`accountMapping.csv`**         | Internal mapping for entities (IDs ↔ organizations).                                                                                         | ❌ Not needed for most ML tasks.                                                                                                  |
| **`individuals-bulkload.csv`**   | Metadata about individual customers (used for graph construction).                                                                           | ⚙️ Optional — use if you’re building a *multi-entity graph* (people + accounts).                                                 |
| **`organizations-bulkload.csv`** | Same as above but for organizations.                                                                                                         | ⚙️ Optional — only for entity-level modeling.                                                                                    |
| **`resolvedentities.csv`**       | Entity resolution info (accounts belonging to the same entity).                                                                              | ⚙️ Optional — useful for aggregating multiple accounts per person/org.                                                           |
| **`tx_log.csv`**                 | Log of generated transactions, similar to `transactions.csv` but more raw.                                                                   | ❌ Usually skip — `transactions.csv` already contains what you need.                                                              |
| **`tx_count.csv`**               | Time-step summary of transaction counts (normal vs SAR).                                                                                     | ⚙️ Optional — for temporal analysis only.                                                                                        |

---

## 🧠 Summary – What You *Actually* Need

If your goal is to train a **money-laundering detection model**, here’s what to use:

### ✅ **Essential Files**

1. `transactions.csv` — main data (edges between accounts)
2. `accounts.csv` — account metadata
3. `sar_accounts.csv` — ground truth suspicious labels
4. `alert_transactions.csv` — transactions labeled as suspicious (optional positives)

### ⚙️ **Optional / Helpful Context**

* `alert_accounts.csv` — for additional suspicious context
* `cash_tx.csv` — if you’re analyzing cash flow typologies
* `resolvedentities.csv` — for entity-level graph aggregation

---

## 🧩 How They Fit Together

Think of the AMLSim dataset as a **graph:**

* **Nodes** → accounts (from `accounts.csv`)
* **Edges** → transactions (from `transactions.csv`)
* **Labels** → SAR / non-SAR (from `sar_accounts.csv` or `isSAR` field in `accounts.csv`)

Then you can:

* Use **`alert_transactions.csv`** as explicit *positive* examples of suspicious behavior.
* Merge with **`transactions.csv`** to create one labeled dataset of all transactions.

---

## 🧪 Typical Workflow Example

If you’re training a GNN or graph-based classifier:

1. **Read accounts:**

   ```python
   accounts = pd.read_csv('accounts.csv')
   ```

2. **Read transactions:**

   ```python
   tx = pd.read_csv('transactions.csv')
   ```

3. **Add labels:**

   ```python
   sar = pd.read_csv('sar_accounts.csv')
   accounts['isSAR'] = accounts['account_id'].isin(sar['account_id']).astype(int)
   ```

4. **Join into edges with node labels:**

   ```python
   edges = tx.merge(accounts[['account_id', 'isSAR']], left_on='orig', right_on='account_id', suffixes=('', '_orig'))
   edges = edges.merge(accounts[['account_id', 'isSAR']], left_on='dest', right_on='account_id', suffixes=('', '_dest'))
   ```

5. **Now you have:**

   * Edge features (amount, type, timestamp)
   * Node features (balances, bank, etc.)
   * Labels (SAR or not)

---

## 🚀 If You Want to Train on a *Single CSV*

You can join transactions + account labels into one combined file, e.g.:

```python
combined = tx.merge(accounts[['account_id', 'isSAR']], left_on='orig', right_on='account_id', how='left')
combined = combined.rename(columns={'isSAR': 'orig_isSAR'})
combined = combined.merge(accounts[['account_id', 'isSAR']], left_on='dest', right_on='account_id', how='left')
combined = combined.rename(columns={'isSAR': 'dest_isSAR'})
combined.to_csv('combined_transactions.csv', index=False)
```

Then you have one dataset suitable for supervised ML.

---

Would you like me to help you create that **merged and labeled combined_transactions.csv** file automatically (a script that builds it cleanly)?
That’s the best next step before training any model.
