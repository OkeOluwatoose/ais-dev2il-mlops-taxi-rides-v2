# AIS DEV2IL 😈 MLOps: Taxi Rides

Welcome to the MLOps Taxi Rides exercises! You are going to explore real-world MLOps practices — 
data management, model training, experiment tracking, and serving predictions — all using a dataset 
of [New York City taxi rides](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

---

## 🛫 Getting Started

1. Fork [this repository](https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2), clone your fork and open it in PyCharm.
2. Make sure you are working on the `main` branch.
3. Install dependencies:
   ```bash
   uv sync
   ```

You're ready to go! 🚀

---

## 📦 Exercise 1: Feel the Pain — CSV vs Parquet

Before we talk about why Parquet is great, let's experience what life looks like without it.

In the `example-data` folder you'll find the same dataset in two formats:
- `2025-01-01.taxi-rides.csv`
- `2025-01-01.taxi-rides.parquet`

Create a new Python file called `compare.py` in the root of the repository and work through the steps below together with your pair.

### Step 1: Load the CSV and inspect the schema

```python
import pandas as pd

df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
print(df_csv.dtypes)
```

Run it with:

```bash
uv run compare.py
```

🤔 Look at the column types carefully:
- What type are `tpep_pickup_datetime` and `tpep_dropoff_datetime`? Are those really the right types for timestamps?
- What type is `ride_time`? It should be a number — but is it?

Open the CSV file in PyCharm and look at the first row. Notice anything odd about `ride_time`?

Someone put `"unknown"` in that field. Because CSV has no schema, pandas silently loaded the entire
column as `object` (string) — **without any warning**. Your data is now silently broken. 😬

### Step 2: Load the Parquet and inspect the schema

```python
df_pq = pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
print(df_pq.dtypes)
```

✅ The datetime columns have the correct type — no extra code needed. The schema is embedded in the file itself.

And that `"unknown"` value? It could never end up in a Parquet file in the first place.
Parquet enforces the column type on **write** — if the data doesn't match, it fails loudly instead of silently corrupting your dataset.

### Step 3: Compare file sizes

```python
import os

csv_size = os.path.getsize('example-data/2025-01-01.taxi-rides.csv')
pq_size  = os.path.getsize('example-data/2025-01-01.taxi-rides.parquet')

print(f'CSV:     {csv_size / 1024 / 1024:.1f} MB')
print(f'Parquet: {pq_size  / 1024 / 1024:.1f} MB')
print(f'Parquet is {csv_size / pq_size:.1f}x smaller')
```

💡 How much disk space would you save if you stored a full year of taxi data as Parquet instead of CSV?

### Step 4: Time the loads

```python
import time

t = time.time()
pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.3f}s')
print(f'Parquet: {pq_time:.3f}s')
```

🗣️ **Discuss with your pair:** You are building an ML pipeline that retrains a model every day.
What problems would these differences cause over time?

### 🚀 Level Up

#### Challenge 1: Chuck Norris Never Needs `pd.to_datetime()`

> *Chuck Norris doesn't parse dates. Dates parse themselves in his presence.*

When you loaded the CSV, the datetime columns came in as plain strings (`object`). Fix them manually to match the Parquet schema:

```python
df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
df_csv['tpep_pickup_datetime']  = pd.to_datetime(df_csv['tpep_pickup_datetime'])
df_csv['tpep_dropoff_datetime'] = pd.to_datetime(df_csv['tpep_dropoff_datetime'])
print(df_csv.dtypes)
```

Now try to fix `ride_time` the same way:

```python
df_csv['ride_time'] = pd.to_numeric(df_csv['ride_time'])
print(df_csv.dtypes)
```

💥 It crashes with a `ValueError`. And that's actually the *right* reaction — you want to train your model on 
valid data. A `ride_time` of `"unknown"` is useless for training and silently coercing it to `NaN` would just hide the problem deeper.

But here's the thing: **you should never have had to deal with this in the first place.** Parquet enforces the schema on 
write — `"unknown"` could never have ended up in a float column. The problem is caught at the source, not discovered halfway 
through your training pipeline.

#### Challenge 2: A Year Worth of Taxi Rides

In ML training, you rarely load just one day of data — you load months or even a full year.
Simulate this by loading the same file 365 times in a loop for both formats:

```python
import time
import pandas as pd

t = time.time()
for _ in range(365):
    pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
for _ in range(365):
    pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.1f}s')
print(f'Parquet: {pq_time:.2f}s')
print(f'Parquet is {csv_time / pq_time:.0f}x faster')
```

⏱️ Every time you retrain your model, you'd be waiting those extra seconds just reading data.
At scale, this adds up fast.

#### Challenge 3: No Code? No Problem!

Explore the Parquet file without writing a single line of Python:

1. In PyCharm's **Project** view, open `example-data/2025-01-01.taxi-rides.parquet`
2. Click **"Edit in Data Wrangler"**
3. Try some transformations:
   - Filter out rows where `outlier` is `True`
   - Sort by `trip_distance` descending
   - Drop the `ride_time` column
4. Click **"Export"** to generate a Python script from your actions
5. Run the generated script with `uv run <script_name>.py`

> Data Wrangler lets you explore visually and then hands you the Python code for free —
> great for getting started with a new dataset fast.
