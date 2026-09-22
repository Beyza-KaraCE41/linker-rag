import os
import ast
import math
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFMCS
import selfies as sf

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


# =========================================================================
# 0. UYARILAR / SELFIES / DEVICE
# =========================================================================

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

custom_constraints = sf.get_semantic_constraints()

custom_constraints.update({
    "N": 5,
    "P": 5,
    "S": 6,
    "I": 7,
    "At": 4
})

for metal in ["Pt", "Zn", "Zr", "Cu", "Fe", "Co", "Ni", "Cd", "Cr",
              "Ag", "Mn", "V", "Ti", "Mo", "W", "Ru", "Rh", "Pd",
              "Os", "Ir", "Au", "Hf", "Al", "Ga", "In"]:
    custom_constraints[metal] = 8

sf.set_semantic_constraints(custom_constraints)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'=' * 80}")
print("M1 MODEL + GRAPHRAG + NO-RAG BASELINE")
print("LEAKAGE-SAFE + MULTI-LINKER + TRAIN-ONLY PREPROCESSING")
print("LatentGenerator -> GRAPHRAG QUERY -> RETRIEVED_Z DECODER")
print(f"{'=' * 80}\n")
print(f"Device: {device}\n")


# =========================================================================
# 1. TEK LINKER CANONICALIZATION
# =========================================================================

def canonicalize_single_smiles(smi):
    if pd.isna(smi):
        return None

    smi = str(smi).strip()

    if not smi:
        return None

    smi = smi.replace("->", "-").replace("<-", "-")

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# =========================================================================
# 2. MULTI-LINKER PARSE
# =========================================================================

def parse_smiles_robust(smi_input):
    """Parse every linker component, including dot-separated CSV cells."""
    if pd.isna(smi_input):
        return None
    try:
        if isinstance(smi_input, (list, tuple, np.ndarray)):
            raw_items = list(smi_input)
        else:
            text = str(smi_input).strip()
            if not text:
                return None
            if text.startswith("["):
                try:
                    parsed = ast.literal_eval(text)
                    raw_items = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
                except Exception:
                    raw_items = [text]
            else:
                raw_items = [text]
        raw_linkers = []
        for item in raw_items:
            if item is None:
                continue
            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass
            raw_linkers.extend([x.strip() for x in str(item).split(".") if x.strip()])
        canonical_linkers = []
        for linker in raw_linkers:
            canonical = canonicalize_single_smiles(linker)
            if canonical is not None:
                canonical_linkers.append(canonical)
        if not canonical_linkers:
            return None
        return ".".join(sorted(set(canonical_linkers)))
    except Exception:
        return None

# =========================================================================
# 3. BİREYSEL LINKERLARI ÇIKAR
# =========================================================================

def extract_individual_linkers(combined_smiles):
    if combined_smiles is None:
        return []
    return [x.strip() for x in str(combined_smiles).split(".") if x.strip()]


# =========================================================================
# 4. GRAPH VALIDATION
# =========================================================================

def is_valid_graph(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


# =========================================================================
# 5. SMILES -> GRAPH
# =========================================================================

def smi_to_graph(smi):
    clean_smi = str(smi).replace("->", "-").replace("<-", "-")

    try:
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None:
            return None

        x = torch.tensor(
            [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge()] + [0] * 12
             for a in mol.GetAtoms()],
            dtype=torch.float32
        )

        edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]

        if edges:
            edge_index = torch.tensor(
                edges + [[j, i] for i, j in edges],
                dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    except Exception:
        return None


# =========================================================================
# 6. SELFIES ENCODING
# =========================================================================

def encode_to_selfies_safe(smi):
    clean_smi = str(smi).replace("->", "-").replace("<-", "-")

    try:
        encoded = sf.encoder(clean_smi)
        if encoded is not None:
            return encoded
    except Exception:
        pass

    components = extract_individual_linkers(clean_smi)

    if len(components) <= 1:
        return None

    encoded_components = []

    for component in components:
        try:
            enc = sf.encoder(component)
            if enc is None:
                return None
            encoded_components.append(enc)
        except Exception:
            return None

    return ".".join(encoded_components)


# =========================================================================
# 7. DATASET HAZIRLAMA
# =========================================================================

def prepare_dataset(filepath):
    print("[INFO] Veri seti yükleniyor...")

    qmof = pd.read_csv(filepath, low_memory=False)

    cols = [
        "info.pld",
        "info.density",
        "outputs.pbe.bandgap",
        "info.symmetry.spacegroup_number",
        "info.symmetry.pointgroup",
        "info.mofid.smiles_linkers"
    ]

    qmof = qmof.dropna(subset=cols).reset_index(drop=True)
    print(f"[INFO] NaN temizliği sonrası: {len(qmof)}")

    qmof["linker_parsed"] = qmof["info.mofid.smiles_linkers"].apply(parse_smiles_robust)
    qmof = qmof.dropna(subset=["linker_parsed"]).reset_index(drop=True)

    qmof["is_valid"] = qmof["linker_parsed"].apply(is_valid_graph)
    qmof = qmof[qmof["is_valid"] == True].reset_index(drop=True)

    print("[INFO] Molecular graphlar hazırlanıyor...")
    qmof["graph"] = qmof["linker_parsed"].apply(smi_to_graph)

    print("[INFO] SELFIES temsilleri hazırlanıyor...")
    qmof["selfies"] = qmof["linker_parsed"].apply(encode_to_selfies_safe)

    qmof = qmof.dropna(subset=["graph", "selfies"]).reset_index(drop=True)

    qmof["linker_components"] = qmof["linker_parsed"].apply(extract_individual_linkers)

    print(f"[INFO] Hazırlanan veri seti: {len(qmof)} örnek")

    num_multi = qmof["linker_components"].apply(len).gt(1).sum()
    print(f"[INFO] Multi-linker kayıt sayısı: {num_multi}")

    unique_linkers = set()
    for linker_list in qmof["linker_components"]:
        for linker in linker_list:
            unique_linkers.add(linker)

    print(f"[INFO] Unique canonical linker bileşeni: {len(unique_linkers)}")

    return qmof


# =========================================================================
# 8. CANONICAL LINKER GROUP SPLIT
# =========================================================================

def split_by_linker_group(df, test_size_needed=400, seed=42):
    df = df.reset_index(drop=True).copy()
    n = len(df)

    row_linkers = [set(x) for x in df["linker_components"]]

    linker_to_rows = {}
    for row_idx, linkers in enumerate(row_linkers):
        for linker in linkers:
            if linker not in linker_to_rows:
                linker_to_rows[linker] = []
            linker_to_rows[linker].append(row_idx)

    visited = set()
    components = []

    for start_idx in range(n):
        if start_idx in visited:
            continue

        stack = [start_idx]
        component = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            for linker in row_linkers[current]:
                for neighbor in linker_to_rows.get(linker, []):
                    if neighbor not in visited:
                        stack.append(neighbor)

        components.append(sorted(component))

    rng = np.random.default_rng(seed)
    rng.shuffle(components)

    selected_components = []
    test_count = 0

    for component in components:
        component_size = len(component)
        if test_count + component_size <= test_size_needed:
            selected_components.append(component)
            test_count += component_size

        if test_count >= test_size_needed:
            break

    if test_count < test_size_needed:
        selected_ids = {id(c) for c in selected_components}
        remaining = [c for c in components if id(c) not in selected_ids]

        if remaining:
            best_component = min(
                remaining,
                key=lambda c: abs(len(c) - (test_size_needed - test_count))
            )
            selected_components.append(best_component)
            test_count += len(best_component)

    test_indices = sorted(idx for component in selected_components for idx in component)
    test_index_set = set(test_indices)
    train_indices = [i for i in range(n) if i not in test_index_set]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    train_linkers = set()
    for linker_list in train_df["linker_components"]:
        for linker in linker_list:
            train_linkers.add(linker)

    test_linkers = set()
    for linker_list in test_df["linker_components"]:
        for linker in linker_list:
            test_linkers.add(linker)

    overlap = train_linkers.intersection(test_linkers)

    if len(overlap) > 0:
        raise RuntimeError(
            f"DATA LEAKAGE TESPİT EDİLDİ! Train/Test ortak canonical linker: {len(overlap)}"
        )

    print(f"[SPLIT] Train: {len(train_df)}")
    print(f"[SPLIT] Test: {len(test_df)}")
    print(f"[SPLIT] Train unique linker: {len(train_linkers)}")
    print(f"[SPLIT] Test unique linker: {len(test_linkers)}")
    print("[SPLIT] Train/Test canonical linker overlap: 0")

    return train_df, test_df


# =========================================================================
# 9. GNN ENCODER
# =========================================================================

class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GATConv(15, 128)
        self.c2 = GATConv(128, 128)

    def forward(self, d):
        h1 = F.leaky_relu(self.c1(d.x, d.edge_index), 0.1)
        h2 = F.leaky_relu(self.c2(h1, d.edge_index), 0.1)
        g = global_mean_pool(h2, d.batch)
        return F.normalize(g, p=2, dim=1)


# =========================================================================
# 10. FORWARD MODEL
# =========================================================================

class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 5)
        )

    def forward(self, z):
        return self.net(z)


# =========================================================================
# 11. LATENT GENERATOR
# =========================================================================

class LatentGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

    def forward(self, p):
        return F.normalize(self.net(p), p=2, dim=1)


# =========================================================================
# 12. POSITIONAL ENCODING
# =========================================================================

class PosEnc(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# =========================================================================
# 13. CONDITIONAL DiT
# =========================================================================

class ConditionalDiT(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()

        self.cf = nn.Sequential(
            nn.Linear(133, 256),
            nn.GELU()
        )

        self.emb = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.pos = PosEnc(256, max_len=max_len)

        self.tr = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                batch_first=True,
                dim_feedforward=1024
            ),
            num_layers=4
        )

        self.hd = nn.Linear(256, vocab_size)

    def forward(self, z, p, t):
        ctx = self.cf(torch.cat([z, p], dim=1)).unsqueeze(1)
        e = self.pos(self.emb(t))

        msk = torch.triu(
            torch.ones(t.size(1), t.size(1), device=t.device) * float("-inf"),
            diagonal=1
        )

        return self.hd(self.tr(tgt=e, memory=ctx, tgt_mask=msk))


# =========================================================================
# 14. SLICED WASSERSTEIN
# =========================================================================

def sliced_wasserstein(x, y, num_projections=50):
    th = torch.randn(x.size(1), num_projections, device=x.device)
    th /= torch.norm(th, dim=0, keepdim=True)

    x_proj = torch.mm(x, th)
    y_proj = torch.mm(y, th)

    x_sorted = torch.sort(x_proj, dim=0)[0]
    y_sorted = torch.sort(y_proj, dim=0)[0]

    return torch.mean(torch.abs(x_sorted - y_sorted))


# =========================================================================
# 15. MOLEKÜLERİ BENZEŞIM SCORE'U İLE KARŞILAŞTIR
# =========================================================================

def calculate_maccs_tanimoto_similarity(smiles1, smiles2):
    """Gerçek 167-bit MACCS Tanimoto. Flexible match'in yalnızca bir bileşenidir."""
    try:
        m1 = Chem.MolFromSmiles(smiles1)
        m2 = Chem.MolFromSmiles(smiles2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = MACCSkeys.GenMACCSKeys(m1)
        fp2 = MACCSkeys.GenMACCSKeys(m2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def calculate_mcs_substructure_score(smiles1, smiles2, timeout=2):
    """
    Esnek yapısal benzerlik.

    MCS (Maximum Common Substructure) ile ortak atom/bond yapısını ölçer.
    Score 0..1 aralığındadır ve iki molekülün de kapsanmasını dikkate alır.
    """
    try:
        m1 = Chem.MolFromSmiles(smiles1)
        m2 = Chem.MolFromSmiles(smiles2)
        if m1 is None or m2 is None:
            return 0.0

        # Tam aynı yapı zaten en güçlü eşleşmedir.
        if Chem.MolToSmiles(m1, canonical=True) == Chem.MolToSmiles(m2, canonical=True):
            return 1.0

        res = rdFMCS.FindMCS(
            [m1, m2],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=timeout
        )

        if res.canceled or res.numAtoms <= 0:
            return 0.0

        atom_den = max(m1.GetNumHeavyAtoms(), m2.GetNumHeavyAtoms(), 1)
        bond_den = max(m1.GetNumBonds(), m2.GetNumBonds(), 1)

        atom_cov = res.numAtoms / atom_den
        bond_cov = res.numBonds / bond_den if res.numBonds > 0 else 0.0

        # Atom ortaklığı + bond/iskelet ortaklığı.
        # Bond yoksa atom coverage tek başına aşırı ödüllendirilmesin.
        if res.numBonds > 0:
            return float(0.60 * atom_cov + 0.40 * bond_cov)
        return float(0.60 * atom_cov)

    except Exception:
        return 0.0


def calculate_substructure_coverage(smiles1, smiles2, timeout=2):
    """MCS'nin iki yapıya ortak kapsama oranını ayrı raporlamak için kullanılır."""
    try:
        m1 = Chem.MolFromSmiles(smiles1)
        m2 = Chem.MolFromSmiles(smiles2)
        if m1 is None or m2 is None:
            return 0.0
        res = rdFMCS.FindMCS(
            [m1, m2],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=timeout
        )
        if res.canceled or res.numAtoms <= 0:
            return 0.0
        return float(res.numAtoms / max(m1.GetNumHeavyAtoms(), m2.GetNumHeavyAtoms(), 1))
    except Exception:
        return 0.0


def _best_one_to_one_component_score(gen_parts, true_parts, pair_score_fn):
    """Multi-linker componentlerini order-independent one-to-one eşleştirir."""
    if not gen_parts or not true_parts:
        return 0.0

    scores = [[float(pair_score_fn(g, t)) for t in true_parts] for g in gen_parts]
    n, m = len(gen_parts), len(true_parts)

    # Küçük component sayılarında DP ile global optimum assignment.
    # Daha büyük listelerde greedy fallback kullanılır.
    if m <= 20:
        from functools import lru_cache

        @lru_cache(None)
        def dp(i, mask):
            if i >= n:
                return 0.0
            best = dp(i + 1, mask)  # generated component unmatched olabilir
            for j in range(m):
                if not (mask & (1 << j)):
                    best = max(best, scores[i][j] + dp(i + 1, mask | (1 << j)))
            return best

        total = dp(0, 0)
    else:
        pairs = sorted(
            ((scores[i][j], i, j) for i in range(n) for j in range(m)),
            reverse=True
        )
        used_i, used_j, total = set(), set(), 0.0
        for sc, i, j in pairs:
            if i not in used_i and j not in used_j:
                used_i.add(i); used_j.add(j); total += sc

    return float(total / max(n, m))


def flexible_linker_similarity(generated_smiles, true_smiles):
    """
    Nihai FLEXIBLE match score.

    1) MACCS Tanimoto
    2) MCS tabanlı ortak yapı skoru
    3) Multi-linker için order-independent component assignment

    Final score = max(MACCS, MCS-structural).
    Bu nedenle exact string equality zorunlu değildir.
    """
    if generated_smiles is None or true_smiles is None:
        return 0.0

    gp = extract_individual_linkers(generated_smiles)
    tp = extract_individual_linkers(true_smiles)
    if not gp or not tp:
        return 0.0

    if len(gp) == 1 and len(tp) == 1:
        maccs = calculate_maccs_tanimoto_similarity(gp[0], tp[0])
        mcs = calculate_mcs_substructure_score(gp[0], tp[0])
        return float(max(maccs, mcs))

    maccs = _best_one_to_one_component_score(
        gp, tp, calculate_maccs_tanimoto_similarity
    )
    mcs = _best_one_to_one_component_score(
        gp, tp, calculate_mcs_substructure_score
    )
    return float(max(maccs, mcs))


def flexible_match_components(generated_smiles, true_smiles):
    """Ayrı component-level flexible similarity."""
    if generated_smiles is None or true_smiles is None:
        return 0.0
    gp = extract_individual_linkers(generated_smiles)
    tp = extract_individual_linkers(true_smiles)
    return _best_one_to_one_component_score(
        gp, tp, flexible_linker_similarity
    )


def calculate_tanimoto_similarity(smiles1, smiles2):
    # Geriye dönük uyumluluk: bu fonksiyon artık esnek linker similarity döndürür.
    return flexible_linker_similarity(smiles1, smiles2)


def calculate_match_metrics(
    generated_smiles_list,
    true_smiles_list,
    thresholds=(0.50, 0.60, 0.70, 0.85, 0.90)
):
    """Flexible structural match + raw MACCS ayrı ayrı raporlanır."""
    flexible_scores = []
    raw_maccs_scores = []
    substructure_scores = []

    for g, t in zip(generated_smiles_list, true_smiles_list):
        if g is None or t is None:
            flexible_scores.append(0.0)
            raw_maccs_scores.append(0.0)
            substructure_scores.append(0.0)
            continue

        gp = extract_individual_linkers(g)
        tp = extract_individual_linkers(t)
        if not gp or not tp:
            flexible_scores.append(0.0)
            raw_maccs_scores.append(0.0)
            substructure_scores.append(0.0)
            continue

        raw_m = _best_one_to_one_component_score(
            gp, tp, calculate_maccs_tanimoto_similarity
        )
        sub_m = _best_one_to_one_component_score(
            gp, tp, calculate_mcs_substructure_score
        )
        flex = max(raw_m, sub_m)

        raw_maccs_scores.append(raw_m)
        substructure_scores.append(sub_m)
        flexible_scores.append(flex)

    out = {}
    for th in thresholds:
        c = sum(x >= th for x in flexible_scores)
        out[th] = {
            "count": int(c),
            "total": len(flexible_scores),
            "ratio": c / len(flexible_scores) if flexible_scores else 0.0
        }

    return out, {
        "flexible_mean": float(np.mean(flexible_scores)) if flexible_scores else 0.0,
        "raw_maccs_mean": float(np.mean(raw_maccs_scores)) if raw_maccs_scores else 0.0,
        "mcs_mean": float(np.mean(substructure_scores)) if substructure_scores else 0.0
    }

# =========================================================================
# 16. SELFIES -> CANONICAL SMILES (MULTI-LINKER [SEP] SUPPORT)
# =========================================================================

def decode_generated_selfies(selfies_str, i2s):
    """
    SELFIES token'larını SMILES'a çevir.

    [SEP] token'ları "." karakterine geri çevrilir ve
    multi-linker yapı korunur.
    """
    if not selfies_str:
        return None

    try:
        selfies_components = [
            c for c in selfies_str.split("[SEP]") if c
        ]

        if len(selfies_components) == 1:
            decoded = sf.decoder(selfies_components[0])
            if decoded:
                return decoded
        else:
            decoded_parts = []

            for component in selfies_components:
                decoded_part = sf.decoder(component)
                if not decoded_part:
                    return None
                decoded_parts.append(decoded_part)

            return ".".join(decoded_parts)

        return None

    except Exception:
        return None


# =========================================================================
# 17. GENERATION + METRİK YARDIMCI FONKSİYONLARI
# =========================================================================

def generate_smiles_from_latent(z, prop_tensor, dit, s2i, i2s, max_len):
    """Verilen latent z ve property vektörü ile decoder'dan autoregressive
    SMILES üretir. Ground-truth kullanılmaz."""
    generated_tokens = [s2i["[BOS]"]]

    for step in range(max_len - 1):
        with torch.no_grad():
            t_in = torch.tensor(
                [generated_tokens],
                dtype=torch.long
            ).to(device)

            out = dit(z, prop_tensor, t_in)

        next_token = out[0, -1, :].argmax().item()

        if next_token == s2i["[SEP]"]:
            generated_tokens.append(next_token)
            continue

        generated_tokens.append(next_token)

        if next_token == s2i["[EOS]"]:
            break

    selfies_str = "".join(
        [i2s.get(t, "") for t in generated_tokens
         if t not in (
             s2i["[PAD]"],
             s2i["[BOS]"],
             s2i["[EOS]"]
         )]
    )

    try:
        gen_smi = decode_generated_selfies(selfies_str, i2s)

        if gen_smi is None:
            return None

        gen_mol = Chem.MolFromSmiles(gen_smi)

        if gen_mol is None:
            return None

        return Chem.MolToSmiles(gen_mol, canonical=True)

    except Exception:
        return None


def evaluate_property_error(gen_smi, target_props_tensor, gnn, fwd):
    """Üretilen molekülü tekrar graph'a çevirip GNN + ForwardModel ile
    property tahmini yapar ve target property ile L2 hatasını döner."""
    graph = smi_to_graph(gen_smi)

    if graph is None:
        return None

    try:
        with torch.no_grad():
            bg = Batch.from_data_list([graph]).to(device)
            z = gnn(bg)
            pred_props = fwd(z)
            err = torch.norm(
                pred_props - target_props_tensor,
                p=2
            ).item()

        return err

    except Exception:
        return None


def compute_validity_uniqueness_novelty(generated_smiles_list, train_canonical_set, train_component_set=None):
    total = len(generated_smiles_list)
    valid = [s for s in generated_smiles_list if s is not None]
    validity = len(valid) / total if total else 0.0
    unique_set = set(valid)
    uniqueness = len(unique_set) / len(valid) if valid else 0.0
    novelty = (len([s for s in unique_set if s not in train_canonical_set]) / len(unique_set)) if unique_set else 0.0
    component_novelty = 0.0
    if train_component_set is not None:
        gen_components = {c for s in unique_set for c in extract_individual_linkers(s)}
        component_novelty = (len(gen_components - train_component_set) / len(gen_components)) if gen_components else 0.0
    return validity, uniqueness, novelty, component_novelty

# =========================================================================
# 18. GRAPHRAG RETRIEVAL
# =========================================================================


# =========================================================================
# 18. VARIANT-SPECIFIC RETRIEVAL UTILITIES
# =========================================================================

def _build_train_bank_metadata(train_df):
    return {
        "smiles": train_df["linker_parsed"].tolist(),
        "components": train_df["linker_components"].tolist(),
    }


def _candidate_metrics_from_indices(query, bank_latents, ids, p, gnn, fwd,
                                    dit, s2i, i2s, max_len):
    candidates = []
    for bi in ids:
        rz = bank_latents[bi].unsqueeze(0)
        with torch.no_grad():
            pred = fwd(rz)
            pe = torch.norm(pred - p, p=2).item()
            dist = torch.norm(rz - query, p=2).item()
        smi = generate_smiles_from_latent(rz, p, dit, s2i, i2s, max_len)
        if smi is None:
            continue
        ae = evaluate_property_error(smi, p, gnn, fwd)
        if ae is None:
            continue
        candidates.append((float(ae), float(dist), smi, int(bi)))
    return candidates


def _select_from_candidates(candidates, final_k):
    if not candidates:
        return [], None, None
    # Candidate tuples: property_error, retrieval_distance, smiles, bank_id.
    front = []
    for i, c in enumerate(candidates):
        dominated = False
        for j, o in enumerate(candidates):
            if i == j:
                continue
            if (o[0] <= c[0] and o[1] <= c[1]
                    and (o[0] < c[0] or o[1] < c[1])):
                dominated = True
                break
        if not dominated:
            front.append(c)
    pool = front if front else candidates
    e = np.asarray([c[0] for c in pool], dtype=np.float32)
    d = np.asarray([c[1] for c in pool], dtype=np.float32)
    en = (e-e.min())/(e.max()-e.min()) if e.max() > e.min() else np.zeros_like(e)
    dn = (d-d.min())/(d.max()-d.min()) if d.max() > d.min() else np.zeros_like(d)
    order = np.argsort(en + dn)
    ranked = [pool[int(i)] for i in order[:final_k]]
    best = ranked[0] if ranked else None
    return ranked, (best[2] if best else None), (best[0] if best else None)


def _evaluate_variant_outputs(generated, truths, errors, train_canonical_set,
                              train_component_set, match_thresholds, extra=None):
    a = compute_validity_uniqueness_novelty(
        generated, train_canonical_set, train_component_set
    )
    m, ms = calculate_match_metrics(generated, truths, match_thresholds)
    out = {
        "validity": a[0],
        "uniqueness": a[1],
        "novelty": a[2],
        "component_novelty": a[3],
        "property_error_mean": float(np.mean(errors)) if errors else None,
        "maccs_similarity_mean": ms["raw_maccs_mean"],
        "mcs_similarity_mean": ms["mcs_mean"],
        "flexible_similarity_mean": ms["flexible_mean"],
        "match_counts": m,
        "n": len(truths),
    }
    if extra:
        out.update(extra)
    return out


def _leave_one_out_retrieval_score(train_latents, train_smiles, mode,
                                   param, gnn=None, fwd=None):
    """
    Training-only hyperparameter selection.
    The queried training sample is explicitly excluded from its own retrieval.
    Score is structural similarity to the known training target, so test labels
    are never used for tuning.
    """
    n = len(train_latents)
    if n < 3:
        return -float("inf")
    sims = []
    with torch.no_grad():
        for i in range(n):
            q = train_latents[i]
            dist = torch.norm(train_latents - q.unsqueeze(0), dim=1)
            dist[i] = float("inf")
            k = min(int(param), n - 1)
            ids = torch.argsort(dist)[:k].tolist()
            if not ids:
                continue
            # Retrieval quality is evaluated against the actual training target,
            # but only for hyperparameter selection inside the training set.
            best = max(
                flexible_linker_similarity(
                    train_smiles[i], train_smiles[j]
                ) for j in ids
            )
            sims.append(best)
    return float(np.mean(sims)) if sims else -float("inf")


def optimize_retrieval_k(train_latents, train_smiles, candidate_ks):
    scores = {}
    for k in candidate_ks:
        scores[int(k)] = _leave_one_out_retrieval_score(
            train_latents, train_smiles, "k", int(k)
        )
    best_k = max(scores, key=scores.get)
    print(f"[TUNING] Retrieval K candidates={list(scores.keys())}")
    print(f"[TUNING] LOO structural retrieval scores={scores}")
    print(f"[TUNING] Selected K={best_k}")
    return int(best_k), scores


# -------------------------------------------------------------------------
# GraphRAG
# -------------------------------------------------------------------------
def build_latent_knn_graph(bank_latents, graph_k, chunk_size=1024):
    """
    Training-only molecular latent graph with memory-safe blockwise kNN.
    The full NxN similarity matrix is never materialized.
    """
    n = bank_latents.size(0)
    k = min(int(graph_k), max(1, n - 1))
    rows = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            block = bank_latents[start:start + chunk_size]
            sim = torch.matmul(block, bank_latents.t())
            local_rows = torch.arange(sim.size(0), device=sim.device)
            local_cols = start + local_rows
            sim[local_rows, local_cols] = -float("inf")
            rows.append(torch.topk(sim, k=k, dim=1).indices)
    return torch.cat(rows, dim=0)


def graph_enhanced_query(query, bank_latents, knn_graph, hops):
    """
    Local graph aggregation. Each retrieved seed contributes its latent plus
    its graph-neighborhood centroid; repeated hops expand the local context.
    """
    with torch.no_grad():
        dist = torch.norm(bank_latents-query, dim=1)
        seed_count = min(int(knn_graph.size(1)), bank_latents.size(0))
        seeds = torch.argsort(dist)[:seed_count].tolist()

        contexts = []
        for idx in seeds:
            current = bank_latents[idx].unsqueeze(0)
            visited = {int(idx)}
            frontier = [int(idx)]
            for _ in range(int(hops)):
                nxt = []
                for node in frontier:
                    for nb in knn_graph[node].tolist():
                        nb = int(nb)
                        if nb not in visited:
                            visited.add(nb)
                            nxt.append(nb)
                if not nxt:
                    break
                frontier = nxt
            nodes = sorted(visited)
            contexts.append(bank_latents[nodes].mean(dim=0))

        if contexts:
            graph_q = torch.stack(contexts, dim=0).mean(dim=0, keepdim=True)
        else:
            graph_q = query
        return F.normalize(graph_q, p=2, dim=1)


def run_graphrag(test_df, feat_cols, gnn, gen, dit, fwd,
                 decoder_latents_tensor, s2i, i2s, max_len,
                 train_canonical_set, train_component_set,
                 K, retrieval_pool, graph_k, hops, match_thresholds):
    gnn.eval(); gen.eval(); dit.eval(); fwd.eval()

    # Graph index is built exclusively from training bank.
    knn_graph = build_latent_knn_graph(decoder_latents_tensor, graph_k)

    top1_g, top1_t, top1_e = [], [], []
    topk_g, topk_t, topk_e = [], [], []
    failed_inference_count = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                        desc="GraphRAG Inference"):
        try:
            p = torch.tensor(
                row[feat_cols].values.astype(np.float32),
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            true = row["linker_parsed"]

            with torch.no_grad():
                gen_z = gen(p)
                graph_q = graph_enhanced_query(
                    gen_z, decoder_latents_tensor, knn_graph, hops
                )
                dist = torch.norm(decoder_latents_tensor-graph_q, dim=1)
                pool_n = min(int(retrieval_pool), len(decoder_latents_tensor))
                ids = torch.argsort(dist)[:pool_n].cpu().tolist()

            rz = decoder_latents_tensor[ids[0]].unsqueeze(0)
            smi = generate_smiles_from_latent(rz, p, dit, s2i, i2s, max_len)
            top1_g.append(smi); top1_t.append(true)
            # PROPERTY_ERROR INDEXING FIX: her satir icin kendi error'i (None dahil)
            e = evaluate_property_error(smi, p, gnn, fwd) if smi is not None else None
            top1_e.append(e)

            candidates = _candidate_metrics_from_indices(
                graph_q, decoder_latents_tensor, ids, p, gnn, fwd, dit,
                s2i, i2s, max_len
            )
            ranked, best_smi, best_err = _select_from_candidates(candidates, K)
            topk_g.append(best_smi); topk_t.append(true)
            if best_err is not None: topk_e.append(best_err)

        except (RuntimeError, ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
            failed_inference_count += 1
            print(f"[INFERENCE ERROR] GraphRAG row idx={row.name}: {type(e).__name__}: {str(e)[:100]}")
            top1_g.append(None); top1_t.append(row.get("linker_parsed", None))
            topk_g.append(None); topk_t.append(row.get("linker_parsed", None))
        except Exception as e:
            failed_inference_count += 1
            print(f"[UNEXPECTED INFERENCE ERROR] GraphRAG row idx={row.name}: {type(e).__name__}: {str(e)[:100]}")
            top1_g.append(None); top1_t.append(row.get("linker_parsed", None))
            topk_g.append(None); topk_t.append(row.get("linker_parsed", None))

    print(f"[INFO] GraphRAG Inference: {failed_inference_count}/{len(test_df)} satır başarısız oldu")

    a = _evaluate_variant_outputs(
        top1_g, top1_t, top1_e, train_canonical_set, train_component_set,
        match_thresholds
    )
    b = _evaluate_variant_outputs(
        topk_g, topk_t, topk_e, train_canonical_set, train_component_set,
        match_thresholds, {"K": K, "retrieval_pool": retrieval_pool,
                           "graph_k": graph_k, "graph_hops": hops}
    )
    # ===== HAM INFERENCE (duzeltildi: girinti + gercek MCS + dogru degiskenler) =====
    rows = []
    for i in range(len(top1_t)):
        true_smi = top1_t[i]
        gen_smi = top1_g[i]
        if gen_smi is None or true_smi is None:
            maccs_i, mcs_i, flex_i, valid_i = None, None, 0.0, False
        else:
            gp = extract_individual_linkers(gen_smi)
            tp = extract_individual_linkers(true_smi)
            maccs_i = _best_one_to_one_component_score(gp, tp, calculate_maccs_tanimoto_similarity)
            mcs_i = _best_one_to_one_component_score(gp, tp, calculate_mcs_substructure_score)
            flex_i = max(maccs_i, mcs_i)
            valid_i = is_valid_graph(gen_smi)
        novel_i = bool(gen_smi and valid_i and gen_smi not in train_canonical_set)
        rows.append({
            "seed": seed, "method": "GraphRAG", "test_index": i,
            "true_linker_smiles": true_smi,
            "generated_linker_smiles": gen_smi,
            "maccs_similarity": maccs_i, "mcs_similarity": mcs_i,
            "flexible_score": flex_i,
            "match_0.50": flex_i >= 0.50, "match_0.60": flex_i >= 0.60,
            "match_0.70": flex_i >= 0.70, "match_0.80": flex_i >= 0.80,
            "match_0.85": flex_i >= 0.85, "match_0.90": flex_i >= 0.90,
            "is_valid": valid_i, "is_novel": novel_i,
            "property_error_l2": top1_e[i] if i < len(top1_e) else None,
            "k": K, "pool": retrieval_pool, "epoch_trained": 145,
        })
    import os as _os
    _outdir = _os.environ.get("OUTDIR", "/kaggle/working")
    pd.DataFrame(rows).to_csv(_os.path.join(_outdir, f"raw_inference_GRAPHRAG_seed{seed}.csv"), index=False)
    # ===== END HAM INFERENCE =====

    return {"top1": a, "topk": b}

# =========================================================================
# 18B. TRAIN-ONLY HYPERPARAMETER OPTIMIZATION
# =========================================================================

def _sample_train_indices(n, max_items=512, seed=42):
    if n <= max_items:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_items, replace=False))


# [KALDIRILDI] tune_retrieval_pool_and_k(): TRAIN-ONLY LOO tuning artık
# kullanılmıyor. K / RETRIEVAL_POOL, seed başına SEED_PARAMS ile sabitlendi
# (bkz. MAIN bölümü). Tuning'in kaldırılması her seed başına ~20-30 dk
# kazandırır.




def tune_graph_parameters(train_latents, train_smiles, seed):
    """
    GraphRAG graph parameters are selected from training-only leave-one-out
    structural retrieval quality. No test label participates.
    """
    idxs = _sample_train_indices(len(train_smiles), seed=seed)
    graph_k_candidates = sorted(set([
        2, 4, 8, 16, 32, min(64, len(train_smiles)-1)
    ]))
    graph_k_candidates = [k for k in graph_k_candidates if k < len(train_smiles)]
    hop_candidates = [1, 2, 3]

    scores = {}
    with torch.no_grad():
        for gk in graph_k_candidates:
            graph = build_latent_knn_graph(train_latents, gk)
            for hops in hop_candidates:
                local = []
                for i in idxs:
                    q = train_latents[i:i+1]
                    # Exclude the queried node by setting its distance to inf
                    # before graph traversal.
                    d = torch.norm(train_latents-q, dim=1)
                    d[i] = float("inf")
                    seeds = torch.argsort(d)[:gk].tolist()
                    visited = set()
                    frontier = [int(x) for x in seeds]
                    for x in frontier:
                        visited.add(x)
                    for _ in range(hops):
                        nxt = []
                        for node in frontier:
                            for nb in graph[node].tolist():
                                nb = int(nb)
                                if nb != i and nb not in visited:
                                    visited.add(nb)
                                    nxt.append(nb)
                        frontier = nxt
                        if not frontier:
                            break
                    if visited:
                        # Similarity-weighted graph context without a manually
                        # chosen mixture coefficient.
                        ids = list(visited)
                        sims = torch.matmul(
                            train_latents[ids], q.squeeze(0)
                        )
                        w = torch.softmax(sims, dim=0)
                        gq = F.normalize(
                            torch.sum(train_latents[ids] * w.unsqueeze(1), dim=0, keepdim=True),
                            p=2, dim=1
                        )
                        rd = torch.norm(train_latents-gq, dim=1)
                        rd[i] = float("inf")
                        j = int(torch.argmin(rd).item())
                        local.append(
                            flexible_linker_similarity(
                                train_smiles[i], train_smiles[j]
                            )
                        )
                scores[(gk, hops)] = (
                    float(np.mean(local)) if local else -float("inf")
                )
    best = max(scores, key=scores.get)
    print(f"[TUNING] Graph parameters (TRAIN-LOO): {scores}")
    print(f"[TUNING] Selected graph_k={best[0]}, hops={best[1]}")
    return int(best[0]), int(best[1])

# =========================================================================
# 19. NO-RAG BASELINE: gen(properties) -> DECODER
# =========================================================================

def run_no_rag_baseline(test_df, feat_cols, gnn, gen, dit, fwd, s2i, i2s, max_len,
                        train_canonical_set, train_component_set, match_thresholds=(0.50,0.60,0.70,0.85,0.90)):
    gnn.eval(); gen.eval(); dit.eval(); fwd.eval()
    smi_list,true_list,errs=[],[],[]
    failed_inference_count = 0
    for _,row in tqdm(test_df.iterrows(), total=len(test_df), desc="No-RAG Baseline Inference"):
        try:
            p=torch.tensor(row[feat_cols].values.astype(np.float32),dtype=torch.float32,device=device).unsqueeze(0)
            with torch.no_grad(): z=gen(p)
            smi=generate_smiles_from_latent(z,p,dit,s2i,i2s,max_len)
            smi_list.append(smi); true_list.append(row["linker_parsed"])
            if smi is not None:
                e=evaluate_property_error(smi,p,gnn,fwd)
                if e is not None:errs.append(e)
        except (RuntimeError, ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
            failed_inference_count += 1
            print(f"[INFERENCE ERROR] No-RAG row idx={row.name}: {type(e).__name__}: {str(e)[:100]}")
            smi_list.append(None); true_list.append(row.get("linker_parsed",None))
        except Exception as e:
            failed_inference_count += 1
            print(f"[UNEXPECTED INFERENCE ERROR] No-RAG row idx={row.name}: {type(e).__name__}: {str(e)[:100]}")
            smi_list.append(None); true_list.append(row.get("linker_parsed",None))
    print(f"[INFO] No-RAG Baseline Inference: {failed_inference_count}/{len(test_df)} satır başarısız oldu")
    a=compute_validity_uniqueness_novelty(smi_list,train_canonical_set,train_component_set)
    m,ms=calculate_match_metrics(smi_list,true_list,match_thresholds)
    return {"validity":a[0],"uniqueness":a[1],"novelty":a[2],"component_novelty":a[3],"property_error_mean":float(np.mean(errs)) if errs else None,"maccs_similarity_mean":ms["raw_maccs_mean"],"mcs_similarity_mean":ms["mcs_mean"],"flexible_similarity_mean":ms["flexible_mean"],"match_counts":m,"n":len(test_df)}


# =========================================================================
# 20. MAIN
# =========================================================================

if __name__ == "__main__":
    csv_path = "/kaggle/input/datasets/beyzanurkara/qmof-dts/qmof.csv"

    if not os.path.exists(csv_path):
        print(f"[HATA] Veri seti bulunamadı: {csv_path}")
        raise SystemExit

    raw_df = prepare_dataset(csv_path)

    SEEDS = [42, 123, 999]

    # Her seed için ayrı ayrı belirlenmiş optimal epoch sayısı
    SEED_EPOCHS = {
        42: 145,
        123: 145,
        999: 145
    }

    BATCH = 32

    RAG_VARIANT = "GRAPHRAG"

    # FIXED K/POOL - NO TUNING (GRAPH)
    SEED_PARAMS = {
        42: {"K": 90, "POOL": 380},
        123: {"K": 90, "POOL": 380},
        999: {"K": 90, "POOL": 380}
    }

    MATCH_THRESHOLDS = (0.50, 0.60, 0.70, 0.85, 0.90)

    rag_top1_results = []
    rag_topk_results = []
    no_rag_results = []

    print(f"\n{'=' * 80}")
    print("DENEY BAŞLADI: RAG (top-1/top-K) vs NO-RAG BASELINE")
    print("Best-of-320 ground-truth oracle KALDIRILDI")
    print("+ MULTI-LINKER [SEP] TOKEN + DYNAMIC MAX_LEN")
    print("+ DECODER TRAINING: REAL_Z + GEN_Z + CROSS-SAMPLE RETRIEVED_Z")
    print("+ FLEXIBLE MATCH: MACCS + MCS/SUBSTRUCTURE")
    print("+ RETRIEVAL POOL: 150 -> PARETO -> TOP-15")
    print(f"{'=' * 80}\n")

    for seed in SEEDS:
        print(f"\n>>> KOŞU SEED: {seed} <<<\n")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Seed'e özel optimal epoch sayısı
        EPOCHS = SEED_EPOCHS[seed]

        print(f"[INFO] Seed {seed} için optimal epoch: {EPOCHS}")

        # FROZEN TEST SETI: split sabit seed ile yapilir; 4 yontem ve 3 model-seed'inin
        # TAMAMI ayni 400 molekullu held-out test seti uzerinde degerlendirilir.
        # (Test boyutu 400, epoch/parametreler ORIJINAL degerlerinde - degistirilmedi.)
        FROZEN_SPLIT_SEED = 42
        train_df, test_df = split_by_linker_group(
            raw_df,
            test_size_needed=400,
            seed=FROZEN_SPLIT_SEED
        )

        print(
            f"[INFO] Train: {len(train_df)}, "
            f"Test: {len(test_df)}"
        )

        feat_cols = [
            "info.pld",
            "info.density",
            "outputs.pbe.bandgap",
            "info.symmetry.spacegroup_number",
            "pointgroup_encoded"
        ]

        # =============================================================
        # LABEL ENCODER: SADECE TRAIN'DE FIT
        # =============================================================

        le = LabelEncoder()

        train_df["pointgroup_encoded"] = (
            le.fit_transform(
                train_df["info.symmetry.pointgroup"]
            ).astype(np.float32) + 1.0
        )

        pointgroup_to_id = {
            label: idx + 1
            for idx, label in enumerate(le.classes_)
        }

        test_df["pointgroup_encoded"] = (
            test_df["info.symmetry.pointgroup"]
            .map(pointgroup_to_id)
            .fillna(0)
            .astype(np.float32)
        )

        # =============================================================
        # STANDARD SCALER: SADECE TRAIN'DE FIT
        # =============================================================

        scaler = StandardScaler()

        train_df[feat_cols] = scaler.fit_transform(
            train_df[feat_cols]
        )

        test_df[feat_cols] = scaler.transform(
            test_df[feat_cols]
        )

        # =============================================================
        # TOKEN VOCABULARY
        # =============================================================

        tokens_set = set()

        for s in train_df["selfies"]:
            for component in str(s).split("."):
                if component:
                    tokens_set.update(
                        sf.split_selfies(component)
                    )

        s2i = {
            token: i + 4
            for i, token in enumerate(sorted(tokens_set))
        }

        s2i.update({
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[SEP]": 3
        })

        i2s = {
            i: token
            for token, i in s2i.items()
        }

        vocab_size = len(s2i)

        # =============================================================
        # DYNAMIC MULTI-LINKER MAX_LEN
        # =============================================================

        def _total_selfies_token_len(selfies_string):
            components = [
                c for c in str(selfies_string).split(".")
                if c
            ]

            total = sum(
                len(list(sf.split_selfies(c)))
                for c in components
            )

            # Her iki component arasına bir [SEP]
            if len(components) > 1:
                total += len(components) - 1

            # [BOS] + content + [EOS]
            return total + 2

        max_train_len = max(
            _total_selfies_token_len(s)
            for s in train_df["selfies"]
        )

        # Gerçek train maksimumunu kullan:
        # multi-linker yapının sonunu MAX_LEN yüzünden kesme.
        MAX_LEN = max_train_len

        print(
            f"[INFO] Vocab: {vocab_size}, "
            f"DYNAMIC MAX_LEN: {MAX_LEN}"
        )

        # =============================================================
        # TOKEN PREPARATION: "." -> [SEP]
        # =============================================================

        def make_training_tokens(selfies_string):
            """
            SELFIES string'i token'lara dönüştürür.

            Multi-linker:
            linker1.linker2.linker3

            ->
            [BOS] linker1 [SEP] linker2 [SEP] linker3 [EOS]
            """

            all_token_ids = []

            components = [
                c for c in str(selfies_string).split(".")
                if c
            ]

            for i, component in enumerate(components):

                for token in sf.split_selfies(component):
                    all_token_ids.append(
                        s2i.get(token, 0)
                    )

                if i < len(components) - 1:
                    all_token_ids.append(
                        s2i["[SEP]"]
                    )

            token_ids = (
                [s2i["[BOS]"]]
                + all_token_ids
                + [s2i["[EOS]"]]
            )

            # Bu MAX_LEN zaten train maksimumu olduğu için
            # normalde truncation oluşmaz.
            if len(token_ids) > MAX_LEN:
                token_ids = (
                    token_ids[:MAX_LEN - 1]
                    + [s2i["[EOS]"]]
                )

            return torch.tensor(
                token_ids,
                dtype=torch.long
            )

        train_df["tokens"] = train_df["selfies"].apply(
            make_training_tokens
        )

        train_df["complexity"] = train_df["tokens"].apply(len)

        train_df = (
            train_df.sort_values(
                by="complexity",
                ascending=True
            )
            .reset_index(drop=True)
        )

        all_graphs = train_df["graph"].tolist()

        all_props = torch.tensor(
            np.array(
                train_df[feat_cols].values,
                dtype=np.float32
            )
        ).to(device)

        all_tokens = train_df["tokens"].tolist()

        # =============================================================
        # MODELS
        # =============================================================

        gnn = GNNEncoder().to(device)
        fwd = ForwardModel().to(device)
        gen = LatentGenerator().to(device)
        dit = ConditionalDiT(
            vocab_size=vocab_size,
            max_len=MAX_LEN
        ).to(device)

        opt = optim.AdamW(
            list(gnn.parameters())
            + list(fwd.parameters())
            + list(gen.parameters())
            + list(dit.parameters()),
            lr=0.0003
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=EPOCHS
        )

        print(
            f"[INFO] Eğitim başlanıyor "
            f"({EPOCHS} epoch)...\n"
        )

        # =============================================================
        # TRAINING
        # =============================================================

        failed_batches_count = 0

        for ep in tqdm(
            range(EPOCHS),
            desc=f"Seed {seed}"
        ):
            gnn.train()
            fwd.train()
            gen.train()
            dit.train()

            if ep < 25:
                subset_size = int(
                    len(all_graphs) * 0.5
                )
            elif ep < 50:
                subset_size = int(
                    len(all_graphs) * 0.75
                )
            else:
                subset_size = len(all_graphs)

            indices = np.random.permutation(
                subset_size
            )

            for i in range(
                0,
                subset_size,
                BATCH
            ):
                try:
                    batch_idx = indices[
                        i:i + BATCH
                    ]

                    bg = Batch.from_data_list(
                        [
                            all_graphs[j]
                            for j in batch_idx
                        ]
                    ).to(device)

                    bp = all_props[batch_idx]

                    bt = pad_sequence(
                        [
                            all_tokens[j]
                            for j in batch_idx
                        ],
                        batch_first=True,
                        padding_value=0
                    ).to(device)

                    if ep < 100:
                        halogen_mask = (
                            (bg.x[:, 0] == 9)
                            | (bg.x[:, 0] == 17)
                            | (bg.x[:, 0] == 35)
                            | (bg.x[:, 0] == 53)
                        )

                        if halogen_mask.any():

                            hal_indices = torch.where(
                                halogen_mask
                            )[0]

                            mask_count = int(
                                len(hal_indices) * 0.15
                            )

                            if mask_count > 0:

                                selected = hal_indices[
                                    torch.randperm(
                                        len(hal_indices)
                                    )[:mask_count]
                                ]

                                bg.x[selected, :] = 0.0

                    # =================================================
                    # REAL LATENT
                    # =================================================

                    real_z = gnn(bg)

                    # =================================================
                    # PROPERTY -> IDEAL LATENT
                    # =================================================

                    gen_z = gen(bp)

                    # =================================================
                    # LATENT ALIGNMENT
                    # =================================================

                    logits = (
                        torch.matmul(
                            gen_z,
                            real_z.t()
                        ) * 14.28
                    )

                    labels = torch.arange(
                        len(bp),
                        device=device
                    )

                    lc = (
                        F.cross_entropy(
                            logits,
                            labels
                        )
                        + F.cross_entropy(
                            logits.t(),
                            labels
                        )
                    ) / 2

                    # =================================================
                    # FORWARD MODEL
                    # =================================================

                    lf = F.mse_loss(
                        fwd(real_z),
                        bp
                    )

                    # =================================================
                    # DISTRIBUTION ALIGNMENT
                    # =================================================

                    sw_loss = sliced_wasserstein(
                        gen_z,
                        real_z
                    )

                    # =================================================
                    # GRAPHRAG RETRIEVAL-CONDITIONED TRAINING
                    # =================================================
                    # Batch-local latent graph: every property query receives
                    # a similarity-normalized message from other training
                    # examples. No hand-written 0.xx mixture coefficient is
                    # introduced; softmax normalization defines the graph
                    # aggregation.
                    if len(bp) > 1:
                        with torch.no_grad():
                            sim = torch.matmul(gen_z, real_z.t())
                            sim.fill_diagonal_(-float("inf"))
                            graph_weights = torch.softmax(sim, dim=1)
                            retrieved_z_train = torch.matmul(
                                graph_weights, real_z
                            ).detach()
                    else:
                        retrieved_z_train = gen_z.detach()

                    dl_real = dit(real_z.detach(), bp, bt[:, :-1])
                    dl_gen = dit(gen_z.detach(), bp, bt[:, :-1])
                    dl_retrieved = dit(retrieved_z_train, bp, bt[:, :-1])


                    target_tokens = bt[:, 1:].reshape(-1)
                    ld_real = F.cross_entropy(
                        dl_real.reshape(-1, vocab_size), target_tokens, ignore_index=0
                    )
                    ld_gen = F.cross_entropy(
                        dl_gen.reshape(-1, vocab_size), target_tokens, ignore_index=0
                    )
                    ld_retrieved = F.cross_entropy(
                        dl_retrieved.reshape(-1, vocab_size), target_tokens, ignore_index=0
                    )

                    # Equal decoder exposure across real/gen/retrieved latents.
                    ld = (ld_real + ld_gen + ld_retrieved) / 3.0

                    # =================================================
                    # TOTAL LOSS
                    # =================================================

                    # Equal weighting of the three principal objectives.
                    # SW remains only a small regularizer.
                    loss = (
                        (1.0 / 3.0) * lc
                        + (1.0 / 3.0) * lf
                        + (1.0 / 3.0) * ld
                        + 0.05 * sw_loss
                    )

                    opt.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        dit.parameters(),
                        max_norm=1.0
                    )

                    opt.step()

                except (RuntimeError, ValueError, TypeError) as e:
                    print(f"[BATCH ERROR] Epoch {ep}, Batch start {i}: {type(e).__name__}: {str(e)[:100]}")
                    failed_batches_count += 1
                    continue
                except torch.cuda.OutOfMemoryError:
                    print(f"[CRITICAL] CUDA Out of Memory at Epoch {ep}, Batch start {i}")
                    torch.save(
                        {
                            "gnn": gnn.state_dict(),
                            "fwd": fwd.state_dict(),
                            "gen": gen.state_dict(),
                            "dit": dit.state_dict(),
                        },
                        f"checkpoint_before_oom_seed{seed}_epoch{ep}.pt"
                    )
                    raise
                except KeyboardInterrupt:
                    print(f"[INFO] Training interrupted by user at Epoch {ep}")
                    raise
                except Exception as e:
                    print(f"[UNEXPECTED ERROR] Epoch {ep}, Batch start {i}: {type(e).__name__}: {str(e)[:100]}")
                    failed_batches_count += 1
                    continue

            scheduler.step()

        print(f"[INFO] Seed {seed}: toplam başarısız batch sayısı = {failed_batches_count}")

        # =============================================================
        # DECODER RETRIEVAL BANK
        # =============================================================

        print(
            "\n[INFO] Decoder bank latentleri oluşturuluyor..."
        )

        gnn.eval()

        decoder_bank_latents = []

        with torch.no_grad():

            for i in range(
                0,
                len(all_graphs),
                BATCH
            ):

                bg = Batch.from_data_list(
                    all_graphs[i:i + BATCH]
                ).to(device)

                decoder_bank_latents.append(
                    gnn(bg).detach()
                )

        decoder_latents_tensor = torch.cat(
            decoder_bank_latents,
            dim=0
        )

        print(
            "[INFO] Latent bank boyutu: "
            f"{decoder_latents_tensor.shape}"
        )

        print(
            "\n[INFO] Test çıkarımı yapılıyor "
            "(RAG top-1/top-K)...\n"
        )

        # =============================================================
        # TRAIN NOVELTY BANK
        # =============================================================

        train_canonical_set = set(train_df["linker_parsed"].tolist())
        train_component_set = set()
        for linker_list in train_df["linker_components"]:
            train_component_set.update(linker_list)

        # =============================================================
        # RAG
        # =============================================================

        # =============================================================
        # TRAIN-ONLY RETRIEVAL HYPERPARAMETER OPTIMIZATION
        # =============================================================
        train_bank_smiles = train_df["linker_parsed"].tolist()

        FINAL_K = SEED_PARAMS[seed]["K"]
        RETRIEVAL_POOL = SEED_PARAMS[seed]["POOL"]

        GRAPH_K, GRAPH_HOPS = tune_graph_parameters(
            decoder_latents_tensor,
            train_bank_smiles,
            seed
        )

        print(
            f"[INFO] {RAG_VARIANT}: FIXED K={FINAL_K}, "
            f"POOL={RETRIEVAL_POOL} (No tuning), "
            f"graph_k={GRAPH_K}, graph_hops={GRAPH_HOPS}"
        )

        # =============================================================
        # GraphRAG
        # =============================================================
        rag_metrics = run_graphrag(
            test_df,
            feat_cols,
            gnn,
            gen,
            dit,
            fwd,
            decoder_latents_tensor,
            s2i,
            i2s,
            MAX_LEN,
            train_canonical_set,
            train_component_set,
            K=FINAL_K,
            retrieval_pool=RETRIEVAL_POOL,
            graph_k=GRAPH_K,
            hops=GRAPH_HOPS,
            match_thresholds=MATCH_THRESHOLDS
        )

        # =============================================================
        # NO-RAG
        # =============================================================

        print(
            "\n[INFO] No-RAG baseline çıkarımı "
            "yapılıyor (gen(properties) -> decoder)...\n"
        )

        no_rag_metrics = run_no_rag_baseline(
            test_df,
            feat_cols,
            gnn,
            gen,
            dit,
            fwd,
            s2i,
            i2s,
            MAX_LEN,
            train_canonical_set,
            train_component_set
        )

        rag_top1_results.append(
            rag_metrics["top1"]
        )

        rag_topk_results.append(
            rag_metrics["topk"]
        )

        no_rag_results.append(
            no_rag_metrics
        )

        def _fmt(m):

            pe = (
                f"{m['property_error_mean']:.4f}"
                if m["property_error_mean"] is not None
                else "N/A"
            )

            mt=m.get("match_counts", {})
            match_070 = mt.get(0.70, {}).get("count", 0)
            return (f"Validity=%{m['validity']*100:.2f}  "
                    f"Uniqueness=%{m['uniqueness']*100:.2f}  "
                    f"Novelty=%{m['novelty']*100:.2f}  "
                    f"ComponentNovelty=%{m.get('component_novelty',0)*100:.2f}  "
                    f"Match@0.70={match_070}/{m.get('n', len(test_df))}  "
                    f"MACCS-Mean={m.get('maccs_similarity_mean',0):.4f}  "
                    f"MCS-Mean={m.get('mcs_similarity_mean',0):.4f}  "
                    f"Flexible-Mean={m.get('flexible_similarity_mean',0):.4f}  "
                    f"PropertyError(L2)={pe}")

        print(
            f"\n[SONUÇ] Seed {seed}:"
        )

        print(
            "  RAG (top-1)     : "
            f"{_fmt(rag_metrics['top1'])}"
        )

        print(
            f"  RAG (pool-{rag_metrics['topk'].get('retrieval_pool', 150)} -> top-{rag_metrics['topk']['K']}) : "
            f"{_fmt(rag_metrics['topk'])}"
        )

        print(
            "  No-RAG baseline : "
            f"{_fmt(no_rag_metrics)}\n"
        )

    # =============================================================
    # FINAL SUMMARY
    # =============================================================

    def _summ(results, key):

        vals = [
            r[key] * 100
            for r in results
        ]

        return (
            np.mean(vals),
            np.std(vals)
        )

    def _summ_err(results):

        errs = [
            r["property_error_mean"]
            for r in results
            if r["property_error_mean"] is not None
        ]

        if not errs:
            return None, None

        return (
            np.mean(errs),
            np.std(errs)
        )

    print(f"\n{'=' * 80}")

    print(
        "SONUÇLAR: "
        "RAG (top-1) vs RAG (top-K) "
        "vs NO-RAG BASELINE"
    )

    print(
        "(Best-of-320 ground-truth oracle KALDIRILDI)"
    )

    print(
        f"{'=' * 80}"
    )

    for label, results in [
        (
            "RAG (top-1)",
            rag_top1_results
        ),
        (
            (
                f"RAG (top-{rag_topk_results[0]['K']})"
                if rag_topk_results
                else "RAG (top-K)"
            ),
            rag_topk_results
        ),
        (
            "No-RAG baseline "
            "(gen(properties) -> decoder)",
            no_rag_results
        ),
    ]:

        val_m, val_s = _summ(
            results,
            "validity"
        )

        uniq_m, uniq_s = _summ(
            results,
            "uniqueness"
        )

        nov_m, nov_s = _summ(results, "novelty")
        cnov_m, cnov_s = _summ(results, "component_novelty")

        err_m, err_s = _summ_err(
            results
        )

        print(
            f"\n{label}:"
        )

        print(
            f"  Validity   : "
            f"%{val_m:.2f} (±%{val_s:.2f})"
        )

        print(
            f"  Uniqueness : "
            f"%{uniq_m:.2f} (±%{uniq_s:.2f})"
        )

        print(
            f"  Novelty    : "
            f"%{nov_m:.2f} (±%{nov_s:.2f})"
        )

        print(f"  Component Novelty: %{cnov_m:.2f} (±%{cnov_s:.2f})")
        for th in (0.50, 0.60, 0.70, 0.85, 0.90):
            counts=[r.get("match_counts",{}).get(th,{}).get("count",0) for r in results]
            print(f"  Match@{th:.2f}: {np.mean(counts):.2f} / {len(test_df)}")

        if err_m is not None:

            print(
                "  Property Error "
                f"(L2, ort.): "
                f"{err_m:.4f} "
                f"(±{err_s:.4f})"
            )

        else:

            print(
                "  Property Error "
                "(L2): N/A"
            )

    # =========================================================================
    # CSV EXPORT: TANIMOTO MATCH RESULTS
    # =========================================================================

    print("\n[CSV EXPORT] Tanimoto match results kaydediliyor...\n")

    csv_data = []

    for label, results in [
        ("RAG (top-1)", rag_top1_results),
        (
            (
                f"RAG (top-{rag_topk_results[0]['K']})"
                if rag_topk_results
                else "RAG (top-K)"
            ),
            rag_topk_results,
        ),
        ("No-RAG baseline (gen(properties) -> decoder)", no_rag_results),
    ]:

        for th in MATCH_THRESHOLDS:
            match_counts = [
                r.get("match_counts", {}).get(th, {}).get("count", 0)
                for r in results
            ]
            match_count_mean = np.mean(match_counts)
            match_count_std = np.std(match_counts)
            match_ratio = match_count_mean / len(test_df) * 100
            match_ratio_std = match_count_std / len(test_df) * 100

            csv_data.append(
                {
                    "Method": label,
                    "Threshold": th,
                    "Match_Count_Mean": round(match_count_mean, 2),
                    "Match_Count_Std": round(match_count_std, 2),
                    "Match_Ratio_%": round(match_ratio, 2),
                    "Std_%": round(match_ratio_std, 2),
                }
            )

    csv_df = pd.DataFrame(csv_data)
    csv_path = "/kaggle/working/graphrag_tanimoto_matches.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"[✓] CSV kaydedildi: {csv_path}\n")
    print(csv_df.to_string(index=False))

    # =========================================================================
    # PNG EXPORT: TANIMOTO VISUALIZATION
    # =========================================================================

    print(f"\n{'=' * 80}")
    print("PNG VISUALIZATION")
    print(f"{'=' * 80}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Match ratios by threshold
    ax = axes[0]
    x_pos = np.arange(len(MATCH_THRESHOLDS))
    width = 0.25

    methods_list = [
        ("RAG (top-1)", rag_top1_results),
        (
            (
                f"RAG (top-{rag_topk_results[0]['K']})"
                if rag_topk_results
                else "RAG (top-K)"
            ),
            rag_topk_results,
        ),
        ("No-RAG baseline", no_rag_results),
    ]

    for i, (label, results) in enumerate(methods_list):
        match_ratios = []
        for th in MATCH_THRESHOLDS:
            match_counts = [
                r.get("match_counts", {}).get(th, {}).get("count", 0)
                for r in results
            ]
            ratio = np.mean(match_counts) / len(test_df) * 100
            match_ratios.append(ratio)

        ax.bar(x_pos + i * width, match_ratios, width, label=label)

    ax.set_xlabel("Tanimoto Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Match Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_title("GraphRAG: Tanimoto Match Ratios by Threshold", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(MATCH_THRESHOLDS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Mean similarity comparison
    ax = axes[1]
    methods_names = [m[0] for m in methods_list]
    mean_sims = []

    for label, results in methods_list:
        # NOT: "mean_similarity" adında bir alan hiçbir zaman üretilmiyordu
        # (result dict'lerinde bu key yoktu) -> her zaman 0.0 dönüyordu.
        # Doğru alan "flexible_similarity_mean" (MACCS + MCS max'inin ortalaması).
        sims = [
            r.get("flexible_similarity_mean", 0.0)
            for r in results
            if r.get("flexible_similarity_mean") is not None
        ]
        mean_sim = np.mean(sims) if sims else 0.0
        mean_sims.append(mean_sim)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.bar(methods_names, mean_sims, color=colors)
    ax.set_ylabel("Mean Flexible Similarity (MACCS/MCS)", fontsize=12, fontweight="bold")
    ax.set_title("GraphRAG: Mean Similarity Across Seeds", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    png_path = "/kaggle/working/graphrag_tanimoto_visualization.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[✓] PNG kaydedildi: {png_path}\n")

    print(
        f"\n{'=' * 80}\n"
    )