

import pandas as pd

dx   = pd.read_csv("models/clinical_model/data/All_Subjects_DXSUM_12Dec2025.csv")
demo = pd.read_csv("models/clinical_model/data/All_Subjects_PTDEMOG_12Dec2025.csv")
mmse = pd.read_csv("models/clinical_model/data/All_Subjects_MMSE_12Dec2025.csv")
cdr  = pd.read_csv("models/clinical_model/data/All_Subjects_CDR_12Dec2025.csv")
amy_meta = pd.read_csv("models/clinical_model/data/All_Subjects_AMYMETA_12Dec2025.csv")
tau_meta = pd.read_csv("models/clinical_model/data/All_Subjects_TAUMETA_12Dec2025.csv")
amyqc= pd.read_csv("models/clinical_model/data/All_Subjects_AMYQC_12Dec2025.csv")

amy_suvr = pd.read_csv("models/clinical_model/data/All_Subjects_UCBERKELEY_AMY_6MM_24Dec2025.csv")
tau_suvr = pd.read_csv("models/clinical_model/data/All_Subjects_UCBERKELEY_TAU_6MM_24Dec2025.csv")
tau_pvc  = pd.read_csv("models/clinical_model/data/All_Subjects_UCBERKELEY_TAUPVC_6MM_24Dec2025.csv")


dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx = dx.dropna(subset=["RID","EXAMDATE"]).sort_values(["RID","EXAMDATE"])
dx_bl = dx.groupby("RID", as_index=False).first()

baseline = dx_bl[["RID","EXAMDATE","DIAGNOSIS"]].copy()
baseline.head()

import pandas as pd

def merge_closest(base, other, other_date_col, prefix, max_days=365):
    """
    Attach the row in 'other' closest in time to base.EXAMDATE for each RID.
    Keeps exactly one row per RID from 'other'. Prefixes ALL columns from other
    (including the date column) to avoid collisions across multiple merges.
    """
    base2 = base[["RID","EXAMDATE"]].rename(columns={"EXAMDATE":"BASE_DATE"}).copy()
    base2["BASE_DATE"] = pd.to_datetime(base2["BASE_DATE"], errors="coerce")

    other2 = other.copy()
    other2[other_date_col] = pd.to_datetime(other2[other_date_col], errors="coerce")
    other2 = other2.dropna(subset=["RID", other_date_col])

    m = other2.merge(base2, on="RID", how="inner")
    m["ABS_DAYS"] = (m[other_date_col] - m["BASE_DATE"]).abs().dt.days
    if max_days is not None:
        m = m[m["ABS_DAYS"] <= max_days]

    m = m.sort_values(["RID","ABS_DAYS"]).groupby("RID", as_index=False).first()

    # Drop helper columns
    m = m.drop(columns=["BASE_DATE","ABS_DAYS"], errors="ignore")

    # Prefix EVERYTHING except RID
    ren = {c: f"{prefix}__{c}" for c in m.columns if c != "RID"}
    m = m.rename(columns=ren)

    return base.merge(m, on="RID", how="left")

def date_candidates(df):
    return [c for c in df.columns if "DATE" in c.upper() or "SCAND" in c.upper() or "VIS" in c.upper()]


AMY_DATE = "SCANDATE" if "SCANDATE" in amy_suvr.columns else date_candidates(amy_suvr)[0]
TAU_DATE = "SCANDATE" if "SCANDATE" in tau_suvr.columns else date_candidates(tau_suvr)[0]
PVC_DATE = "SCANDATE" if "SCANDATE" in tau_pvc.columns else date_candidates(tau_pvc)[0]


master = baseline.copy()

# demographics
demo2 = demo.drop_duplicates("RID").copy()
demo2 = demo2.rename(columns={c: f"DEMO__{c}" for c in demo2.columns if c != "RID"})
master = master.merge(demo2, on="RID", how="left")

# cognitive (VISDATE)
mmse["VISDATE"] = pd.to_datetime(mmse["VISDATE"], errors="coerce")
cdr["VISDATE"]  = pd.to_datetime(cdr["VISDATE"], errors="coerce")

master = merge_closest(master, mmse, "VISDATE", "MMSE", max_days=180)
master = merge_closest(master, cdr,  "VISDATE", "CDR",  max_days=180)

# PET metadata + QC (SCANDATE)
if "SCANDATE" in amy_meta.columns:
    master = merge_closest(master, amy_meta, "SCANDATE", "AMYMETA", max_days=365)
if "SCANDATE" in tau_meta.columns:
    master = merge_closest(master, tau_meta, "SCANDATE", "TAUMETA", max_days=365)
if "SCANDATE" in amyqc.columns:
    master = merge_closest(master, amyqc, "SCANDATE", "AMYQC", max_days=365)

# PET scores (UCBerkeley) — use detected date cols (likely SCANDATE)
master = merge_closest(master, amy_suvr, AMY_DATE, "AMY_SUVR", max_days=365)
master = merge_closest(master, tau_suvr, TAU_DATE, "TAU_SUVR", max_days=365)
master = merge_closest(master, tau_pvc,  PVC_DATE, "TAU_PVC",  max_days=365)

amy_cols = [c for c in master.columns if c.startswith("AMY_SUVR__")]
tau_cols = [c for c in master.columns if c.startswith("TAU_SUVR__")]
pvc_cols = [c for c in master.columns if c.startswith("TAU_PVC__")]

def pick_biomarker(cols, keywords):
    for k in keywords:
        hits = [c for c in cols if k in c.upper()]
        if hits:
            return hits[0]
    return None

AMY_GLOBAL = pick_biomarker(amy_cols, ["CENTILOID","SUMMARY","GLOBAL","COMPOSITE","MEAN","META"])
TAU_META   = pick_biomarker(tau_cols, ["META","COMPOSITE","TEMPORAL","SUMMARY","GLOBAL"])
TAU_PVC_META = pick_biomarker(pvc_cols, ["META","COMPOSITE","TEMPORAL","SUMMARY","GLOBAL"])


master["AMY_BIOMARKER"] = master[AMY_GLOBAL] if AMY_GLOBAL else pd.NA
master["TAU_BIOMARKER"] = master[TAU_META] if TAU_META else pd.NA
master["TAU_PVC_BIOMARKER"] = master[TAU_PVC_META] if TAU_PVC_META else pd.NA

master["MMSE_TOTAL"] = master.get("MMSE__MMSCORE")
master["CDR_SB"] = master.get("CDR__CDRSB")
master["CDR_GLOBAL"] = master.get("CDR__CDGLOBAL")

master[["RID","DIAGNOSIS","MMSE_TOTAL","CDR_SB","AMY_BIOMARKER","TAU_BIOMARKER","TAU_PVC_BIOMARKER"]].head(10)

key_cols = ["MMSE_TOTAL","CDR_SB","AMY_BIOMARKER","TAU_BIOMARKER","TAU_PVC_BIOMARKER"]


amy_df = master.dropna(subset=["AMY_BIOMARKER"]).copy()

amy_df["DIAGNOSIS"].value_counts().sort_index()

dx_map = {
    1: "CN",
    2: "MCI",
    3: "AD"
}

amy_df["DX_LABEL"] = amy_df["DIAGNOSIS"].map(dx_map)
amy_df["DX_LABEL"].value_counts()

import pandas as pd

model_df = amy_df[[
    "AMY_BIOMARKER",
    "MMSE_TOTAL",
    "CDR_SB",
    "DEMO__PTGENDER",
    "DEMO__PTEDUCAT",
    "DX_LABEL"
]].copy()

# Correct ADNI gender mapping
model_df["SEX"] = model_df["DEMO__PTGENDER"].map({
    1.0: 0,  # Male
    2.0: 1   # Female
})

# Education as numeric
model_df["EDUC"] = pd.to_numeric(model_df["DEMO__PTEDUCAT"], errors="coerce")

# Drop rows with missing values (very few)
final = model_df[[
    "AMY_BIOMARKER",
    "MMSE_TOTAL",
    "CDR_SB",
    "SEX",
    "EDUC",
    "DX_LABEL"
]].dropna()


amy_tau = amy_df.dropna(subset=["TAU_BIOMARKER"]).copy()

model_tau = amy_tau[[
    "AMY_BIOMARKER",
    "TAU_BIOMARKER",          # <-- tau added
    "MMSE_TOTAL",
    "CDR_SB",
    "DEMO__PTGENDER",
    "DEMO__PTEDUCAT",
    "DX_LABEL"
]].copy()

# ADNI sex coding: 1=Male, 2=Female
model_tau["SEX"] = model_tau["DEMO__PTGENDER"].map({1.0: 0, 2.0: 1})
model_tau["EDUC"] = pd.to_numeric(model_tau["DEMO__PTEDUCAT"], errors="coerce")

final_tau = model_tau[[
    "AMY_BIOMARKER",
    "TAU_BIOMARKER",
    "MMSE_TOTAL",
    "CDR_SB",
    "SEX",
    "EDUC",
    "DX_LABEL"
]].dropna()

final_tau['DX_LABEL'] = final_tau['DX_LABEL'].map({'CN':0,'MCI':1,'AD':2})

prediction_classes = {
    0: 'CN',
    1: 'MCI',
    2: 'AD'
}

from xgboost import XGBClassifier

X = final_tau.iloc[:,:-1]
print(X.head())
y = final_tau.iloc[:,-1]
print(y.head())

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15/0.85,
    random_state=42,
    stratify=y_temp
)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",   # or "merror"
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=30,
    verbose_eval=50
)



bst.save_model("models/clinical_model/data/alzheimers_clinical_model.json")
final_tau.to_csv("models/clinical_model/data/final_tau.csv", index=False)








