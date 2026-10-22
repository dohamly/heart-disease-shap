"""
Le dashboard (web/index.html) recalcule la prédiction et SHAP en JavaScript quand
l'API est hors ligne. Ce test garantit que ce calcul reste IDENTIQUE à celui de l'API :
si quelqu'un réentraîne le modèle sans relancer `python src/export_dashboard.py`,
la CI échoue au lieu de laisser le dashboard afficher des chiffres périmés.

Nécessite Node.js (présent sur les runners GitHub) ; le test est ignoré sinon.
"""

import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js absent")

JS = r"""
const fs=require('fs'), vm=require('vm');
const html=fs.readFileSync(process.argv[1],'utf8');
const src=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const el=()=>({innerHTML:'',textContent:'',value:'',style:{},dataset:{},classList:{toggle(){},contains(){return false}},addEventListener(){}});
const ctx={document:{getElementById:el,querySelector:el,querySelectorAll:()=>[],createElement:el},
  fetch:()=>Promise.reject(new Error('offline')),AbortController:class{abort(){} get signal(){return {}}},
  setTimeout:()=>0,clearTimeout:()=>0,console};
vm.createContext(ctx); vm.runInContext(src,ctx);
const patients=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(patients.map(p=>{const r=ctx.computeLocally(p);
  return {proba:r.proba,base:r.base,baseProba:r.baseProba,space:r.space,byVar:ctx.aggregateByVariable(r.contrib)};})));
"""


def _patients(n=25):
    df = pd.read_csv(os.path.join(ROOT, "data", "heart.csv"))
    df = df[(df.ca <= 3) & (df.thal >= 1)].drop(columns="target").sample(n, random_state=3)
    return [{k: (float(v) if k == "oldpeak" else int(v)) for k, v in r.items()} for r in df.to_dict("records")]


def test_dashboard_javascript_matches_the_api():
    patients = _patients()
    run = subprocess.run(["node", "-e", JS, os.path.join(ROOT, "web", "index.html")],
                         input=json.dumps(patients), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[:500]
    local = json.loads(run.stdout)

    client = TestClient(app)
    for p, loc in zip(patients, local):
        api = client.post("/explain", json=p).json()
        assert abs(loc["proba"] - api["prediction"]["probability_disease"]) < 2e-4
        assert api["output_space"] == loc["space"] == "log-odds"
        assert abs(loc["base"] - api["base_value"]) < 2e-4
        assert abs(loc["baseProba"] - api["base_probability"]) < 2e-4
        by_var = {c["feature"]: c["value"] for c in api["variable_contributions"]}
        assert set(by_var) == set(loc["byVar"])
        assert max(abs(by_var[k] - loc["byVar"][k]) for k in by_var) < 2e-4
