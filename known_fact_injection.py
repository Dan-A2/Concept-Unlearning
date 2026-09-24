"""Known-fact injection with an EXACT oracle (WMDP-adjacent synthetic setting).

The point: build a case where the ground truth is exact by construction, using
two separate adapters, and check that TRIAGE calls exact removal "localized"
(large positive AdjGap), not "collateral-dominant".

Entities: 40 fictitious researchers, 3 attributes each, one fixed template.
Disjoint matched groups F (20 entities / 60 facts) and A (20 / 60): same
template, same domain, different entities -> A is genuinely adjacent. ~2 probes
per fact (cloze + 4-option MCQ) -> ~240 probes. Base must be at chance.

Injection with two separate adapters, both restricted to layer range L (default
12-15):
    theta_inj    = base + adapter_A + adapter_F   ("knows both")
    theta_oracle = base + adapter_A               (exact removal of F)

Stages (default: all of 1-4):
  1 corpora  build + save facts/probes                         (cpu)
  2 train    adapter_A on A, adapter_F on F (layers L only)    (gpu)
  3 verify   accuracies: base@chance, inj>90% both, oracle A>90% F@chance
  4 triage   CoFi/CHess shift(theta_inj -> theta_oracle) on C_F=F, C_A=A,
             C_G=WikiText; AdjGap = dCoFi(C_F) - dCoFi(C_A); localization mass
             inside vs outside L. THE validity check.
  * save_inj save theta_inj + theta_oracle as full HF models and export F/A as
             on-disk datasets (data/inject-{F,A}) so the baseline unlearners can
             run stage 5.

Stage 5 (audit real methods vs the oracle) has two steps:
  (a) produce unlearned-from-theta_inj checkpoints -- run each baseline with the
      new 'inject' benchmark:
        python ga.py  --benchmark inject \
            --model_path checkpoints/injection/<model>_inj --model_name inj-ga
        (same for rmu.py / npo.py / atu.py; writes checkpoints/<m>/inject/<name>)
  (b) audit them vs the oracle:
        python known_fact_injection.py --model <model> --stages audit \
            --unlearned ga::checkpoints/ga/inject/inj-ga rmu::... npo::... atu::...
      -> per method: shift(theta_inj -> method) on C_F/C_A/C_G, localization
         mass inside L, and retained A-probe accuracy.

Outputs under Results/rebuttal/injection/.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

import exp_common as X
import test as T

OUT = X.PROJECT_ROOT / "Results" / "rebuttal" / "injection"
ADAPTER_DIR = X.PROJECT_ROOT / "checkpoints" / "injection"

TEMPLATES = {
    "field":     "{name}'s primary research field is {value}.",
    "hometown":  "{name} was born in the city of {value}.",
    "discovery": "{name} is known for discovering the {value} effect.",
}
QUESTION = {
    "field":     "What is {name}'s primary research field?",
    "hometown":  "In which city was {name} born?",
    "discovery": "What effect is {name} known for discovering?",
}
ATTRS = list(TEMPLATES)

# invented, single-token-ish values so the base model is at chance
_SYL = ["zol", "quen", "brax", "vind", "mor", "keth", "ald", "sy", "tor", "wex",
        "phal", "gri", "dun", "yar", "nix", "olm", "cae", "rho", "tiv", "ulf"]


def _rng(seed):
    return random.Random(seed)


def _make_value(r):
    return (r.choice(_SYL) + r.choice(_SYL) + r.choice(["ium", "ara", "ex", "os", "yr"])).capitalize()


def build_corpora(seed=0):
    r = _rng(seed)
    names = []
    while len(names) < 40:
        nm = "Dr. " + (r.choice(_SYL) + r.choice(_SYL)).capitalize() + " " + \
             (r.choice(_SYL) + r.choice(_SYL)).capitalize()
        if nm not in names:
            names.append(nm)
    entities = []
    used = set()
    for nm in names:
        vals = {}
        for a in ATTRS:
            v = _make_value(r)
            while v in used:
                v = _make_value(r)
            used.add(v)
            vals[a] = v
        entities.append({"name": nm, "attrs": vals})
    F = entities[:20]
    A = entities[20:]

    def facts(group):
        return [TEMPLATES[a].format(name=e["name"], value=e["attrs"][a])
                for e in group for a in ATTRS]

    def probes(group):
        pr = []
        for e in group:
            for a in ATTRS:
                correct = e["attrs"][a]
                # distractors: same attribute from 3 other entities
                others = [x["attrs"][a] for x in entities if x is not e]
                r.shuffle(others)
                opts = [correct] + others[:3]
                r.shuffle(opts)
                pr.append({"name": e["name"], "attr": a,
                           "question": QUESTION[a].format(name=e["name"]),
                           "options": opts, "answer": opts.index(correct),
                           "cloze": TEMPLATES[a].format(name=e["name"], value="").rstrip(" ."),
                           "cloze_answer": correct})
        return pr

    data = {"F": {"facts": facts(F), "probes": probes(F), "entities": F},
            "A": {"facts": facts(A), "probes": probes(A), "entities": A}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(OUT / "corpora.json", "w"), indent=2)
    print(f"[corpora] F: {len(data['F']['facts'])} facts / {len(data['F']['probes'])} probes; "
          f"A: {len(data['A']['facts'])} facts / {len(data['A']['probes'])} probes")
    return data


# ---------------------------------------------------------------------------
# MCQ accuracy via option log-likelihood
# ---------------------------------------------------------------------------

@torch.no_grad()
def mcq_accuracy(model, tok, probes):
    dev = T._get_input_device(model)
    correct = 0
    for p in probes:
        stem = p["question"] + " Answer: "
        lls = []
        for opt in p["options"]:
            ids = tok(stem + str(opt), return_tensors="pt").input_ids.to(dev)
            out = model(ids)
            logits = out.logits[0, :-1].float().log_softmax(-1)
            tgt = ids[0, 1:]
            lls.append(logits[range(len(tgt)), tgt].sum().item())
        if int(torch.tensor(lls).argmax()) == p["answer"]:
            correct += 1
    return correct / max(1, len(probes))


# ---------------------------------------------------------------------------
# Layer-restricted LoRA adapter training
# ---------------------------------------------------------------------------

def train_adapter(base_path, facts, layers, out_dir, epochs=8, lr=2e-4, seed=0):
    from peft import LoraConfig, get_peft_model, TaskType
    X.seed_everything(seed)
    model, tok = X.load_model(base_path)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                     lora_dropout=0.0, target_modules=list(T.DEFAULT_TARGET_MODULES),
                     layers_to_transform=list(layers), layers_pattern="layers")
    model = get_peft_model(model, cfg)
    model.train()
    dev = T._get_input_device(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(facts)
        tot = 0.0
        for txt in facts:
            ids = tok(txt, return_tensors="pt").input_ids.to(dev)
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"  [train {out_dir.name}] epoch {ep+1}/{epochs} loss {tot/len(facts):.4f}")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model
    X.free()


def _merge_adapters(base_path, adapter_paths):
    """base + sequentially-merged LoRA adapters -> a full model."""
    from peft import PeftModel
    model, tok = X.load_model(base_path)
    for ap in adapter_paths:
        model = PeftModel.from_pretrained(model, str(ap))
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def save_injection_models(base_path, model_key, ap_A, ap_F, corpora):
    """Save theta_inj + theta_oracle as full HF models, and export the F/A
    facts as on-disk datasets, so the baseline unlearners can run stage 5:

        python <method>.py --benchmark inject \
            --model_path checkpoints/injection/<model>_inj --model_name <label>

    (--benchmark inject resolves forget=inject-F, retain=inject-A via
     baselines/benchmarks.py -> data/inject-{F,A}.)"""
    from datasets import Dataset, DatasetDict
    for grp, facts in (("F", corpora["F"]["facts"]), ("A", corpora["A"]["facts"])):
        dpath = X.PROJECT_ROOT / "data" / f"inject-{grp}"
        DatasetDict({"train": Dataset.from_dict({"text": list(facts)})}).save_to_disk(str(dpath))
        print(f"[save_inj] corpus data/inject-{grp}  ({len(facts)} facts)")

    inj, toki = _merge_adapters(base_path, [ap_A, ap_F])
    p_inj = ADAPTER_DIR / f"{model_key}_inj"
    inj.save_pretrained(str(p_inj)); toki.save_pretrained(str(p_inj))
    print(f"[save_inj] theta_inj -> {p_inj}")
    del inj, toki; X.free()

    orc, tko = _merge_adapters(base_path, [ap_A])
    p_or = ADAPTER_DIR / f"{model_key}_oracle"
    orc.save_pretrained(str(p_or)); tko.save_pretrained(str(p_or))
    print(f"[save_inj] theta_oracle -> {p_or}")
    del orc, tko; X.free()


# ---------------------------------------------------------------------------
# TRIAGE shift on the (inj -> oracle) pair + localization mass
# ---------------------------------------------------------------------------

def _layer_of(param_name):
    import re
    m = re.search(r"layers\.(\d+)\.", param_name)
    return int(m.group(1)) if m else -1


def localization_mass(diag_inj, diag_oracle, layers):
    """Fraction of squared CoFi shift falling inside layer range `layers`."""
    inside = outside = 0.0
    lset = set(layers)
    for k in diag_inj:
        d = (diag_inj[k].double() - diag_oracle[k].double()).pow(2).sum().item()
        if _layer_of(k) in lset:
            inside += d
        else:
            outside += d
    tot = inside + outside
    return inside / tot if tot > 0 else float("nan")


def triage_pair(inj_model, tok_inj, oracle_model, tok_or, corpora, layers, wikitext, out):
    """dCoFi/dCHess shift(inj -> oracle) on C_F=F, C_A=A, C_G=WikiText."""
    tp_inj = X.targets(inj_model)
    tp_or = X.targets(oracle_model)
    texts = {"C_F": corpora["F"]["facts"], "C_A": corpora["A"]["facts"],
             "C_G": wikitext[:200]}
    res = {}
    loc = {}
    for name, tx in texts.items():
        di = X.cofi_diag(inj_model, tok_inj, tx, tp_inj)
        do = X.cofi_diag(oracle_model, tok_or, tx, tp_or)
        res[("cofi", name)] = X.shift(di, do)
        if name == "C_F":
            loc["cofi"] = localization_mass(di, do, layers)
        del di, do
        X.free()
        ci = X.chess_diag(inj_model, tok_inj, tx, tp_inj, X.CHESS_K, base_seed=1, corpus_name=name)
        co = X.chess_diag(oracle_model, tok_or, tx, tp_or, X.CHESS_K, base_seed=2, corpus_name=name)
        res[("chess", name)] = X.shift(ci, co)
        del ci, co
        X.free()

    adjgap_cofi = res[("cofi", "C_F")] - res[("cofi", "C_A")]
    adjgap_chess = res[("chess", "C_F")] - res[("chess", "C_A")]
    for (metric, name), v in res.items():
        out.row(stage="triage_oracle", metric=metric, corpus=name, quantity="shift",
                value=f"{v:.6f}")
    out.row(stage="triage_oracle", metric="cofi", corpus="-", quantity="adjgap", value=f"{adjgap_cofi:.6f}")
    out.row(stage="triage_oracle", metric="chess", corpus="-", quantity="adjgap", value=f"{adjgap_chess:.6f}")
    out.row(stage="triage_oracle", metric="cofi", corpus="C_F", quantity="localization_mass_inL", value=f"{loc['cofi']:.6f}")
    print(f"[triage] AdjGap(CoFi)={adjgap_cofi:.3f}  AdjGap(CHess)={adjgap_chess:.3f}  "
          f"localization_mass_inL(CoFi)={loc['cofi']:.3f}")
    return res


COLUMNS = ["stage", "metric", "corpus", "quantity", "value"]

# Columns that uniquely identify a row (everything except the result itself),
# used to skip work already recorded on disk when resuming.
KEY_COLUMNS = ["stage", "metric", "corpus", "quantity"]


def run(args):
    X.setup_determinism()
    OUT.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    layers = args.layers
    base_path = X.MODELS[args.model]
    csv_path = OUT / f"injection_{args.model}.csv"
    resume = not args.fresh
    done = X.load_done_keys(csv_path, KEY_COLUMNS) if resume else set()
    if done:
        print(f"[{T._ts()}] resuming: {len(done)} row(s) already in {csv_path}")
    out = X.CSVWriter(csv_path, COLUMNS, resume=resume)

    stages = args.stages
    corpora = None
    if "corpora" in stages or (OUT / "corpora.json").exists():
        corpora = (build_corpora(args.seed) if "corpora" in stages
                   else json.load(open(OUT / "corpora.json")))

    if "train" in stages:
        train_adapter(base_path, list(corpora["A"]["facts"]), layers,
                      ADAPTER_DIR / f"{args.model}_A", epochs=args.epochs, seed=args.seed)
        train_adapter(base_path, list(corpora["F"]["facts"]), layers,
                      ADAPTER_DIR / f"{args.model}_F", epochs=args.epochs, seed=args.seed)

    ap_A = ADAPTER_DIR / f"{args.model}_A"
    ap_F = ADAPTER_DIR / f"{args.model}_F"

    if "save_inj" in stages:
        save_injection_models(base_path, args.model, ap_A, ap_F, corpora)

    if "verify" in stages:
        # base @ chance
        base, tokb = X.load_model(base_path)
        for grp in ("F", "A"):
            acc = mcq_accuracy(base, tokb, corpora[grp]["probes"])
            out.row(stage="verify", metric="mcq", corpus=grp, quantity="acc_base", value=f"{acc:.4f}")
            print(f"[verify] base acc {grp} = {acc:.3f} (expect ~0.25)")
        del base, tokb; X.free()
        # oracle = base + A
        orc, tko = _merge_adapters(base_path, [ap_A])
        for grp in ("F", "A"):
            acc = mcq_accuracy(orc, tko, corpora[grp]["probes"])
            out.row(stage="verify", metric="mcq", corpus=grp, quantity="acc_oracle", value=f"{acc:.4f}")
            print(f"[verify] oracle acc {grp} = {acc:.3f} (expect A>0.9, F~0.25)")
        del orc, tko; X.free()
        # inj = base + A + F
        inj, toki = _merge_adapters(base_path, [ap_A, ap_F])
        for grp in ("F", "A"):
            acc = mcq_accuracy(inj, toki, corpora[grp]["probes"])
            out.row(stage="verify", metric="mcq", corpus=grp, quantity="acc_inj", value=f"{acc:.4f}")
            print(f"[verify] inj acc {grp} = {acc:.3f} (expect both >0.9)")
        del inj, toki; X.free()

    if "triage" in stages:
        _, _, retain = X.wmdp_corpora()
        wikitext = X.wmdp_corpora()[0]["retain2"]
        inj, toki = _merge_adapters(base_path, [ap_A, ap_F])
        orc, tko = _merge_adapters(base_path, [ap_A])
        triage_pair(inj, toki, orc, tko, corpora, layers, wikitext, out)
        del inj, toki, orc, tko; X.free()

    if "audit" in stages:
        audit(args, base_path, corpora, layers, out, done)

    out.close()
    print(f"\n[{T._ts()}] wrote {csv_path}")


def audit(args, base_path, corpora, layers, out, done):
    """Stage 5: compare externally-produced unlearned checkpoints to the oracle.
    Each --unlearned entry is LABEL::PATH (a model fine-tuned/unlearned FROM
    theta_inj targeting F). We measure shift(theta_inj -> that model), and
    also theta_inj's own accuracy on F/A -- the reference point every
    audited method's acc[label] is actually being compared against (without
    it, e.g. acc[ga]=0.48 has no baseline to say whether that's a big or
    small drop).

    Resumable: skips any (stage, metric, corpus, quantity) already in `done`,
    at the granularity of individual measurements, so re-running to pick up
    a newly-available checkpoint (or this new baseline) doesn't repeat
    already-computed CoFi diagonals for methods already audited.
    """
    if not args.unlearned:
        print("[audit] no --unlearned checkpoints given; skipping.")
        return
    wikitext = X.wmdp_corpora()[0]["retain2"]
    inj, toki = _merge_adapters(base_path, [ADAPTER_DIR / f"{args.model}_A",
                                            ADAPTER_DIR / f"{args.model}_F"])
    tp_inj = X.targets(inj)
    texts = {"C_F": corpora["F"]["facts"], "C_A": corpora["A"]["facts"], "C_G": wikitext[:200]}
    inj_diag = {n: X.cofi_diag(inj, toki, tx, tp_inj) for n, tx in texts.items()}

    def emit(metric, corpus, quantity, value_str):
        out.row(stage="audit", metric=metric, corpus=corpus, quantity=quantity, value=value_str)
        done.add(("audit", metric, corpus, quantity))

    # theta_inj's own accuracy on F and A -- the baseline every audited
    # method's acc[label] should be read against.
    for grp in ("F", "A"):
        key = ("audit", "mcq", grp, "acc[theta_inj]")
        if key not in done:
            acc = mcq_accuracy(inj, toki, corpora[grp]["probes"])
            emit("mcq", grp, "acc[theta_inj]", f"{acc:.4f}")
            print(f"[audit] theta_inj acc {grp} = {acc:.3f}")

    for spec in args.unlearned:
        label, path = spec.split("::", 1)
        need_shift = {n: ("audit", "cofi", n, f"shift[{label}]") not in done for n in texts}
        need_loc = ("audit", "cofi", "C_F", f"localization_mass_inL[{label}]") not in done
        need_acc = ("audit", "mcq", "A", f"acc[{label}]") not in done
        if not (any(need_shift.values()) or need_loc or need_acc):
            print(f"[audit] {label}: already done, skipping.")
            continue

        m, tok = X.load_model(path)
        tp = X.targets(m)
        for n, tx in texts.items():
            if not (need_shift[n] or (n == "C_F" and need_loc)):
                continue
            d = X.cofi_diag(m, tok, tx, tp)
            if need_shift[n]:
                sh = X.shift(inj_diag[n], d)
                emit("cofi", n, f"shift[{label}]", f"{sh:.6f}")
            if n == "C_F" and need_loc:
                loc = localization_mass(inj_diag[n], d, layers)
                emit("cofi", "C_F", f"localization_mass_inL[{label}]", f"{loc:.6f}")
            del d; X.free()
        if need_acc:
            acc_A = mcq_accuracy(m, tok, corpora["A"]["probes"])
            emit("mcq", "A", f"acc[{label}]", f"{acc_A:.4f}")
        del m, tok; X.free()
    del inj, toki; X.free()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama-3.1-8b", choices=list(X.MODELS))
    p.add_argument("--stages", nargs="+",
                   default=["corpora", "train", "verify", "triage", "save_inj"],
                   help="corpora train verify triage save_inj audit")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unlearned", nargs="*", default=None,
                   help="audit stage: LABEL::PATH of models unlearned from theta_inj.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing injection_<model>.csv and recompute "
                        "everything (default: resume, appending and skipping rows "
                        "already on disk).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
