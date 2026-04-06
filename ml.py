from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnull, avg, count
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                    MulticlassClassificationEvaluator,
                                    RegressionEvaluator)
from pyspark.ml import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────
# 1. SETUP & LOAD
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("FlightDelay_ML") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

OUTPUT_DIR = "/home/hadoopuser/flight_analysis"

df = spark.read.csv(
    "hdfs:///flight_delay/input/input_data.csv",
    header=True,
    inferSchema=True
)

print(f"Total rows loaded: {df.count():,}")

# ─────────────────────────────────────────────
# 2. SAFE FEATURE ENGINEERING (Pre-Flight Only)
# ─────────────────────────────────────────────
# We only keep features that are known BEFORE the flight arrives.
df = df.withColumn("is_weekend",
    when(col("day_of_week").isin("Saturday", "Sunday"), 1.0).otherwise(0.0))

# Target Variable for Classification (1 = delayed 15+ min, 0 = on time)
df = df.withColumn("label",
    when(col("ARR_DEL15") >= 1, 1.0).otherwise(0.0))

df_model = df.filter(
    col("label").isNotNull() &
    col("ARR_DELAY").isNotNull()
)

print(f"Rows after filtering nulls in targets: {df_model.count():,}")

# ─────────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────────
numeric_features = [
    # Core flight timing (Known prior to arrival)
    "DEP_DELAY",              
    "DEP_DELAY_NEW",          
    "CRS_ELAPSED_TIME",       
    "DISTANCE",               

    # Scheduled Turnaround (Known pre-flight)
    "scheduled_Turnarnd",

    # Weather (Treating as forecasts known at departure)
    "max_temp_f",
    "min_temp_f",
    "max_dewpoint_f",
    "min_dewpoint_f",
    "precip_in",
    "avg_wind_speed_kts",
    "snow_in",
    "avg_feel",

    # Engineered (Pre-flight knowns)
    "is_weekend",

    # Time
    "MONTH",
    "DAY_OF_MONTH"
]

cat_features = ["MKT_CARRIER", "OP_CARRIER", "ORIGIN", "DEST", "FAA_class"]
cat_indexed  = [f + "_idx" for f in cat_features]

all_input_cols = numeric_features + cat_features

# Drop rows with missing features
df_clean = df_model.select(
    all_input_cols + ["label", "ARR_DELAY"]
).dropna()

print(f"Rows after dropna on safe features: {df_clean.count():,}")

print("\nClass balance (label = delayed 15+ min):")
df_clean.groupBy("label").count().show()

# ─────────────────────────────────────────────
# 4. ML PIPELINE PREP
# ─────────────────────────────────────────────
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in cat_features
]

assembler = VectorAssembler(
    inputCols=numeric_features + cat_indexed,
    outputCol="features",
    handleInvalid="keep"
)

train, test = df_clean.randomSplit([0.8, 0.2], seed=42)
print(f"\nTrain rows : {train.count():,}")
print(f"Test rows  : {test.count():,}")

# ─────────────────────────────────────────────
# MODEL A: Random Forest Classifier
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL A: Random Forest Classifier — delayed 15+ min?")
print("=" * 60)

rf_clf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=8,
    maxBins=400,  # <--- FIXED: Allows for high-cardinality categorical features
    seed=42
)

pipeline_clf = Pipeline(stages=indexers + [assembler, rf_clf])
print("Training classifier...")
model_clf = pipeline_clf.fit(train)

preds_clf = model_clf.transform(test)

auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(preds_clf)
acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(preds_clf)
f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(preds_clf)

print(f"\n  AUC-ROC            : {auc:.4f}")
print(f"  Accuracy           : {acc:.4f}")
print(f"  F1 Score           : {f1:.4f}")

# Plot Feature Importances
rf_clf_model = model_clf.stages[-1]
feat_names   = numeric_features + cat_indexed
importances  = list(zip(feat_names, rf_clf_model.featureImportances.toArray()))
importances.sort(key=lambda x: x[1], reverse=True)

top15 = importances[:15]
names, vals = zip(*top15)

plt.figure(figsize=(10, 6))
plt.barh(list(reversed(names)), list(reversed(vals)), color="#378ADD")
plt.xlabel("Importance Score")
plt.title("Top 15 Feature Importances — RF Classifier")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_07_feature_importance_clf.png", dpi=150)
plt.close()
print("\n  Saved: plot_07_feature_importance_clf.png")

# ─────────────────────────────────────────────
# MODEL B: Logistic Regression Classifier
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL B: Logistic Regression Classifier (comparison)")
print("=" * 60)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=False, withStd=True)

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

pipeline_lr = Pipeline(stages=indexers + [assembler, scaler, lr])
print("Training logistic regression...")
model_lr = pipeline_lr.fit(train)
preds_lr = model_lr.transform(test)

auc_lr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(preds_lr)
acc_lr = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(preds_lr)
f1_lr = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(preds_lr)

print(f"\n  AUC-ROC  : {auc_lr:.4f}")
print(f"  Accuracy : {acc_lr:.4f}")
print(f"  F1 Score : {f1_lr:.4f}")

# ─────────────────────────────────────────────
# MODEL C: Random Forest Regressor
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL C: Random Forest Regressor — predict ARR_DELAY (minutes)")
print("=" * 60)

train_r, test_r = df_clean.randomSplit([0.8, 0.2], seed=42)

rf_reg = RandomForestRegressor(
    featuresCol="features",
    labelCol="ARR_DELAY",
    numTrees=100,
    maxDepth=8,
    maxBins=400, # <--- FIXED
    seed=42
)

pipeline_reg = Pipeline(stages=indexers + [assembler, rf_reg])
print("Training regressor...")
model_reg = pipeline_reg.fit(train_r)
preds_reg = model_reg.transform(test_r)

rmse = RegressionEvaluator(labelCol="ARR_DELAY", predictionCol="prediction", metricName="rmse").evaluate(preds_reg)
mae = RegressionEvaluator(labelCol="ARR_DELAY", predictionCol="prediction", metricName="mae").evaluate(preds_reg)
r2 = RegressionEvaluator(labelCol="ARR_DELAY", predictionCol="prediction", metricName="r2").evaluate(preds_reg)

print(f"\n  RMSE : {rmse:.2f} minutes")
print(f"  MAE  : {mae:.2f} minutes")
print(f"  R²   : {r2:.4f}")

sample_pdf = preds_reg.select("ARR_DELAY", "prediction") \
    .filter((col("ARR_DELAY") > -60) & (col("ARR_DELAY") < 300)) \
    .sample(fraction=0.01, seed=42) \
    .toPandas()

plt.figure(figsize=(7, 7))
plt.scatter(sample_pdf["ARR_DELAY"], sample_pdf["prediction"], alpha=0.2, s=6, color="#5DCAA5")
plt.plot([-60, 300], [-60, 300], "r--", linewidth=1.2, label="Perfect prediction")
plt.xlabel("Actual ARR_DELAY (min)")
plt.ylabel("Predicted ARR_DELAY (min)")
plt.title("RF Regressor: Actual vs Predicted Arrival Delay")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_08_actual_vs_predicted.png", dpi=150)
plt.close()
print("  Saved: plot_08_actual_vs_predicted.png")

# ─────────────────────────────────────────────
# 5. SUMMARY & EXPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Model':<35} {'AUC':>6} {'Accuracy':>9} {'F1':>7}")
print(f"  {'-'*60}")
print(f"  {'RF Classifier':<35} {auc:>6.4f} {acc:>9.4f} {f1:>7.4f}")
print(f"  {'Logistic Regression':<35} {auc_lr:>6.4f} {acc_lr:>9.4f} {f1_lr:>7.4f}")
print(f"\n  RF Regressor (ARR_DELAY minutes):")
print(f"    RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

models     = ["RF Classifier", "Logistic Regression"]
auc_scores = [auc, auc_lr]
acc_scores = [acc, acc_lr]
f1_scores  = [f1, f1_lr]

x = range(len(models))
width = 0.25

plt.figure(figsize=(9, 5))
plt.bar([i - width for i in x], auc_scores, width, label="AUC-ROC", color="#378ADD")
plt.bar([i         for i in x], acc_scores, width, label="Accuracy", color="#1D9E75")
plt.bar([i + width for i in x], f1_scores,  width, label="F1 Score",  color="#D85A30")
plt.xticks(list(x), models)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Classification Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_09_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: plot_09_model_comparison.png")

print("\nSaving predictions to HDFS...")

preds_clf.select(
    "MONTH", "DAY_OF_MONTH", "MKT_CARRIER", "ORIGIN", "DEST",
    "DEP_DELAY", "label", "prediction"
).write.mode("overwrite").parquet("hdfs:///flight_delay/output/clf_predictions")

preds_reg.select(
    "MONTH", "DAY_OF_MONTH", "MKT_CARRIER", "ORIGIN", "DEST",
    "DEP_DELAY", "ARR_DELAY", "prediction"
).write.mode("overwrite").parquet("hdfs:///flight_delay/output/reg_predictions")

print("Classifier predictions -> hdfs:///flight_delay/output/clf_predictions")
print("Regressor  predictions -> hdfs:///flight_delay/output/reg_predictions")

print("\n" + "=" * 60)
print("ML COMPLETE — check ~/flight_analysis/ for all plots")
print("=" * 60)

spark.stop()