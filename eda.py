from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, avg
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required on server
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder \
    .appName("FlightDelay_EDA") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

OUTPUT_DIR = "/home/hadoopuser/flight_analysis"

df = spark.read.csv(
    "hdfs:///flight_delay/input/input_data.csv",
    header=True,
    inferSchema=True
)

print("=" * 60)
print("SCHEMA")
print("=" * 60)
df.printSchema()

total_rows = df.count()
print(f"\nTotal rows : {total_rows:,}")
print(f"Total cols : {len(df.columns)}\n")

print("=" * 60)
print("NULL COUNTS PER COLUMN")
print("=" * 60)

exprs = []
for c_name, c_type in df.dtypes:
    # Apply isnan() only to Double or Float types
    if c_type in ('double', 'float'):
        exprs.append(count(when(isnull(col(c_name)) | isnan(col(c_name)), col(c_name))).alias(c_name))
    else:
        # For dates, strings, and integers, only check for nulls
        exprs.append(count(when(isnull(col(c_name)), col(c_name))).alias(c_name))

null_counts = df.select(exprs)
null_counts.show(truncate=False, vertical=True)

print("=" * 60)
print("DESCRIPTIVE STATISTICS (numeric columns)")
print("=" * 60)
numeric_cols = [
    "DEP_DELAY", "DEP_DELAY_NEW", "ARR_DELAY", "ARR_DELAY_NEW",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY", "max_temp_f", "min_temp_f",
    "max_dewpoint_f", "min_dewpoint_f", "precip_in",
    "avg_wind_speed_kts", "snow_in", "avg_feel",
    "scheduled_Turnarnd", "Actual_Turnarnd", "Diff_in_turnarnd"
]
df.select(numeric_cols).describe().show(truncate=False)

print("=" * 60)
print("ARR_DELAY — distribution stats")
print("=" * 60)
df.select("ARR_DELAY").describe().show()

print("Delay category breakdown:")
df.groupBy(
    when(col("ARR_DELAY") < 0,   "Early")
    .when(col("ARR_DELAY") <= 15, "On-time (0-15 min)")
    .when(col("ARR_DELAY") <= 60, "Minor delay (15-60 min)")
    .when(col("ARR_DELAY") >  60, "Major delay (>60 min)")
    .alias("category")
).count().orderBy("count", ascending=False).show()

print("=" * 60)
print("AVG ARR_DELAY by day_of_week")
print("=" * 60)
df.groupBy("day_of_week") \
  .agg(
      avg("ARR_DELAY").alias("avg_arr_delay"),
      avg("ARR_DEL15").alias("pct_delayed_15min"),
      count("*").alias("flight_count")
  ) \
  .orderBy("day_of_week") \
  .show()

print("=" * 60)
print("AVG ARR_DELAY by MONTH")
print("=" * 60)
df.groupBy("MONTH") \
  .agg(
      avg("ARR_DELAY").alias("avg_arr_delay"),
      avg("ARR_DEL15").alias("pct_delayed_15min"),
      count("*").alias("flight_count")
  ) \
  .orderBy("MONTH") \
  .show()

print("=" * 60)
print("AVG ARR_DELAY by MKT_CARRIER")
print("=" * 60)
df.groupBy("MKT_CARRIER") \
  .agg(
      avg("ARR_DELAY").alias("avg_arr_delay"),
      avg("ARR_DEL15").alias("pct_delayed_15min"),
      count("*").alias("flight_count")
  ) \
  .orderBy("avg_arr_delay", ascending=False) \
  .show()

print("=" * 60)
print("TOP 20 ORIGINS by avg delay")
print("=" * 60)
df.groupBy("ORIGIN") \
  .agg(
      avg("ARR_DELAY").alias("avg_arr_delay"),
      count("*").alias("flight_count")
  ) \
  .filter(col("flight_count") > 100) \
  .orderBy("avg_arr_delay", ascending=False) \
  .show(20)

print("=" * 60)
print("AVG delay by cause type")
print("=" * 60)
delay_causes = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
df.select([avg(c).alias(c) for c in delay_causes]).show()

print("=" * 60)
print("PEARSON CORRELATION: weather features vs ARR_DELAY")
print("=" * 60)
weather_cols = [
    "max_temp_f", "min_temp_f", "max_dewpoint_f", "min_dewpoint_f",
    "precip_in", "avg_wind_speed_kts", "snow_in", "avg_feel"
]
for wc in weather_cols:
    try:
        c = df.stat.corr("ARR_DELAY", wc)
        print(f"  {wc:30s}: {c:+.4f}")
    except Exception as e:
        print(f"  {wc:30s}: ERROR — {e}")

print("=" * 60)
print("TURNAROUND IMPACT ON DELAYS")
print("=" * 60)
df.select(
    avg("scheduled_Turnarnd").alias("avg_scheduled_turnaround"),
    avg("Actual_Turnarnd").alias("avg_actual_turnaround"),
    avg("Diff_in_turnarnd").alias("avg_turnaround_diff")
).show()

print("Correlation: Diff_in_turnarnd vs ARR_DELAY =",
      df.stat.corr("Diff_in_turnarnd", "ARR_DELAY"))
print("Correlation: longTurnaround vs ARR_DELAY =",
      df.stat.corr("longTurnaround", "ARR_DELAY"))


# -- Plot 1: ARR_DELAY distribution (5% sample) --
pdf_delay = df.select("ARR_DELAY") \
    .filter(col("ARR_DELAY").isNotNull() &
            (col("ARR_DELAY") > -60) &
            (col("ARR_DELAY") < 300)) \
    .sample(fraction=0.05, seed=42) \
    .toPandas()

plt.figure(figsize=(10, 5))
plt.hist(pdf_delay["ARR_DELAY"], bins=100, color="#378ADD",
         edgecolor="white", linewidth=0.3)
plt.axvline(0,  color="red",    linestyle="--", label="On-time (0 min)")
plt.axvline(15, color="orange", linestyle="--", label="15 min threshold")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Count")
plt.title("Distribution of Arrival Delays (5% sample)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_01_delay_distribution.png", dpi=150)
plt.close()
print("Saved: plot_01_delay_distribution.png")

# -- Plot 2: Avg delay by day of week --
dow_pdf = df.groupBy("day_of_week") \
    .agg(avg("ARR_DELAY").alias("avg_delay")) \
    .orderBy("day_of_week") \
    .toPandas()

plt.figure(figsize=(9, 4))
bars = plt.bar(dow_pdf["day_of_week"].astype(str),
               dow_pdf["avg_delay"],
               color=["#E24B4A" if v > 10 else "#1D9E75"
                      for v in dow_pdf["avg_delay"]])
plt.xlabel("Day of Week")
plt.ylabel("Avg Arrival Delay (min)")
plt.title("Average Arrival Delay by Day of Week")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_02_delay_by_dow.png", dpi=150)
plt.close()
print("Saved: plot_02_delay_by_dow.png")

# -- Plot 3: Avg delay by month --
month_pdf = df.groupBy("MONTH") \
    .agg(avg("ARR_DELAY").alias("avg_delay")) \
    .orderBy("MONTH") \
    .toPandas()

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
month_pdf["month_name"] = month_pdf["MONTH"].apply(
    lambda x: month_names[int(x)-1] if pd.notnull(x) else "?"
)

plt.figure(figsize=(10, 4))
plt.plot(month_pdf["month_name"], month_pdf["avg_delay"],
         marker="o", linewidth=2, color="#5DCAA5", markersize=6)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Month")
plt.ylabel("Avg Arrival Delay (min)")
plt.title("Average Arrival Delay by Month")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_03_delay_by_month.png", dpi=150)
plt.close()
print("Saved: plot_03_delay_by_month.png")

# -- Plot 4: Delay cause breakdown --
cause_pdf = df.select(
    [avg(c).alias(c) for c in delay_causes]
).toPandas().T.reset_index()
cause_pdf.columns = ["Cause", "Avg_Minutes"]
cause_pdf = cause_pdf.sort_values("Avg_Minutes", ascending=True)

plt.figure(figsize=(8, 4))
plt.barh(cause_pdf["Cause"], cause_pdf["Avg_Minutes"], color="#D85A30")
plt.xlabel("Avg Delay (minutes, among delayed flights)")
plt.title("Average Contribution by Delay Cause")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_04_delay_causes.png", dpi=150)
plt.close()
print("Saved: plot_04_delay_causes.png")

# -- Plot 5: Weather correlation bar chart --
corr_vals = {}
for wc in weather_cols:
    try:
        corr_vals[wc] = df.stat.corr("ARR_DELAY", wc)
    except Exception:
        pass

corr_pdf = pd.DataFrame(list(corr_vals.items()),
                         columns=["Feature", "Correlation"])
corr_pdf = corr_pdf.sort_values("Correlation")

plt.figure(figsize=(9, 4))
colors = ["#E24B4A" if v < 0 else "#1D9E75"
          for v in corr_pdf["Correlation"]]
plt.barh(corr_pdf["Feature"], corr_pdf["Correlation"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation with ARR_DELAY")
plt.title("Weather Feature Correlations with Arrival Delay")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_05_weather_correlations.png", dpi=150)
plt.close()
print("Saved: plot_05_weather_correlations.png")

# -- Plot 6: DEP_DELAY vs ARR_DELAY scatter (1% sample) --
scatter_pdf = df.select("DEP_DELAY", "ARR_DELAY") \
    .filter(col("DEP_DELAY").isNotNull() &
            col("ARR_DELAY").isNotNull() &
            (col("DEP_DELAY") > -30) & (col("DEP_DELAY") < 300) &
            (col("ARR_DELAY") > -60) & (col("ARR_DELAY") < 300)) \
    .sample(fraction=0.01, seed=42) \
    .toPandas()

plt.figure(figsize=(7, 6))
plt.scatter(scatter_pdf["DEP_DELAY"], scatter_pdf["ARR_DELAY"],
            alpha=0.15, s=6, color="#7F77DD")
plt.plot([-30, 300], [-30, 300], "r--", linewidth=1, label="DEP = ARR")
plt.xlabel("Departure Delay (min)")
plt.ylabel("Arrival Delay (min)")
plt.title("Departure Delay vs Arrival Delay (1% sample)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_06_dep_vs_arr_delay.png", dpi=150)
plt.close()
print("Saved: plot_06_dep_vs_arr_delay.png")

print("\n" + "=" * 60)
print("EDA COMPLETE — check ~/flight_analysis/ for all plots")
print("=" * 60)

spark.stop()