from importlib import import_module
import os
import sys
import argparse
import linecache
import math
import numpy as np
import glob
import json
import uproot
import vector
import matplotlib
import matplotlib.pyplot as plt
import awkward as ak
matplotlib.use('Agg')


vector.register_awkward()





def overlap_removal(target, cleans, cut=0.4, dphi=False):
    mask = ak.ones_like(target["pt"], dtype=bool)
    for clean in cleans:
        pairs = ak.cartesian([target, clean], nested=True)
        # ΔR 계산
        raw = (pairs["0"].deltaphi(pairs["1"]) if dphi else pairs["0"].deltaR(pairs["1"]))
        delta = np.abs(raw)
        # 0인 값(=자기 자신)을 무시하기 위해 np.inf로 대체
        nonzero = ak.where(delta > 0, delta, np.inf)
        # 이제 nonzero 중 최소값을 취함 → 사실상 두 번째로 작은 원래 delta
        min_dr = ak.min(nonzero, axis=2)
        mask = mask & (min_dr > cut)
    return target[mask]

def toptagging_overlap_removal(target, cleans, cut=0.4, dphi=False,events= None):
    toptagging = events["FatJet_particleNetWithMass_TvsQCD"].array()
    mask = ak.ones_like(target["pt"], dtype=bool)
    for clean in cleans:
        pairs = ak.cartesian([target, clean], nested=True)
        # ΔR 계산
        raw = (pairs["0"].deltaphi(pairs["1"]) if dphi else pairs["0"].deltaR(pairs["1"]))
        delta = np.abs(raw)
        # 0인 값(=자기 자신)을 무시하기 위해 np.inf로 대체
        nonzero = ak.where(delta > 0, delta, np.inf)
        # 이제 nonzero 중 최소값을 취함 → 사실상 두 번째로 작은 원래 delta
        min_dr = ak.min(nonzero, axis=2)
        mask = mask & (min_dr > cut)
    return toptagging[mask]

def sdm_toptagging_overlap_removal(target, cleans, cut=0.4, dphi=False, events=None):
    softdrop_mass = events["FatJet_msoftdrop"].array()
    mask = ak.ones_like(target["pt"], dtype=bool)
    for clean in cleans:
        pairs = ak.cartesian([target, clean], nested=True)
        # ΔR 계산
        raw = (pairs["0"].deltaphi(pairs["1"]) if dphi else pairs["0"].deltaR(pairs["1"]))
        delta = np.abs(raw)
        # 0인 값(=자기 자신)을 무시하기 위해 np.inf로 대체
        nonzero = ak.where(delta > 0, delta, np.inf)
        # 이제 nonzero 중 최소값을 취함 → 사실상 두 번째로 작은 원래 delta
        min_dr = ak.min(nonzero, axis=2)
        mask = mask & (min_dr > cut)
    return softdrop_mass[mask]

def btagging_overlap_removal(target, cleans, cut=0.4, dphi=False,events=None):
    btagging = events["FatJet_btagDeepB"].array()
    mask = ak.ones_like(target["pt"], dtype=bool)
    for clean in cleans:
        pairs = ak.cartesian([target, clean], nested=True)
        # ΔR 계산
        raw = (pairs["0"].deltaphi(pairs["1"]) if dphi else pairs["0"].deltaR(pairs["1"]))
        delta = np.abs(raw)
        # 0인 값(=자기 자신)을 무시하기 위해 np.inf로 대체
        nonzero = ak.where(delta > 0, delta, np.inf)
        # 이제 nonzero 중 최소값을 취함 → 사실상 두 번째로 작은 원래 delta
        min_dr = ak.min(nonzero, axis=2)
        mask = mask & (min_dr > cut)
    return btagging[mask]

def overlap_itself_removal(target, cleans, cut=0.4, dphi=False):
            mask = ak.ones_like(target["pt"], dtype=bool)
            for clean in cleans:
                pairs = ak.cartesian([target, clean], nested=True) # axis 0 = #event , axis 1 = target , axis 2 = clean
                delta = np.abs(pairs["0"].deltaphi(pairs["1"]) if dphi else pairs["0"].deltaR(pairs["1"]))
                mask = mask & (ak.min(delta, axis=2) > cut)
            return target[mask]
def find_closest_jet(obj, jets):
    # obj, jets: both are jagged arrays of shape (n_events,), each sublist = Momentum4D
    pairs = ak.cartesian([obj, jets], nested=True)          # shape: (n_events, N_obj, N_jet)
    dr = pairs["0"].deltaR(pairs["1"])                       # same shape
    # 이벤트별로 obj 하나당 가장 작은 ΔR 의 jet index
    closest_idx_per_obj = ak.argmin(dr, axis=2)              # shape = (n_events, N_obj)
    return jets[closest_idx_per_obj]
def btag_find_closest_jet(obj, jets):
    # obj, jets: both are jagged arrays of shape (n_events,), each sublist = Momentum4D
    pairs = ak.cartesian([obj, jets], nested=True)          # shape: (n_events, N_obj, N_jet)
    dr = pairs["0"].deltaR(pairs["1"])                       # same shape
    # 이벤트별로 obj 하나당 가장 작은 ΔR 의 jet index
    closest_idx_per_obj = ak.argmin(dr, axis=2)              # shape = (n_events, N_obj)
    return btagging[closest_idx_per_obj]
def Select(inputcoll,etamax,ptmin) :
    output = []
    for obj in inputcoll :
        if abs(obj["eta"]) < etamax and obj["pt"] > ptmin : output.append(obj)
    return output 


def twoak8cleancut(sample):
    file = uproot.open(sample)
    events = file["Events"]
    keys = events.keys()
    # LHE-level particles
    lhe_pdgid = events["LHEPart_pdgId"].array()
    lhe_pt    = events["LHEPart_pt"].array()
    lhe_eta   = events["LHEPart_eta"].array()
    lhe_phi   = events["LHEPart_phi"].array()
    lhe_mass  = events["LHEPart_mass"].array()

    bottom_mask     = (lhe_pdgid == 5) | (lhe_pdgid == -5)
    lhe_muon_mask   = (lhe_pdgid == 13) | (lhe_pdgid == -13)
    lhe_particle    = ak.zip({
        "pt":   lhe_pt,
        "eta":  lhe_eta,
        "phi":  lhe_phi,
        "mass": lhe_mass
    }, with_name="Momentum4D")

    lhe_bottoms     = lhe_particle[bottom_mask]
    lhe_bottoms_eta  = lhe_bottoms["eta"][:, 1:2]
    lhe_bottoms_phi  = lhe_bottoms["phi"][:, 1:2]
    lhe_bottoms_pt   = lhe_bottoms["pt"][:, 1:2]
    lhe_bottoms_mass = lhe_bottoms["mass"][:, 1:2]

    lhe_bottom2_eta  = lhe_bottoms["eta"][:, 0:1]
    lhe_bottom2_phi  = lhe_bottoms["phi"][:, 0:1]
    lhe_bottom2_pt   = lhe_bottoms["pt"][:, 0:1]
    lhe_bottom2_mass = lhe_bottoms["mass"][:, 0:1]

    lhe_bottom = ak.zip({
        "pt":   lhe_bottoms_pt,
        "eta":  lhe_bottoms_eta,
        "phi":  lhe_bottoms_phi,
        "mass": lhe_bottoms_mass
    }, with_name="Momentum4D")
    lhe_bottom2 = ak.zip({
        "pt":   lhe_bottom2_pt,
        "eta":  lhe_bottom2_eta,
        "phi":  lhe_bottom2_phi,
        "mass": lhe_bottom2_mass
    }, with_name="Momentum4D")

    lhe_muons      = lhe_particle[lhe_muon_mask]
    lhe_muons_pt   = lhe_muons["pt"]
    lhe_muons_eta  = lhe_muons["eta"]
    lhe_muons_phi  = lhe_muons["phi"]
    lhe_muons_mass = lhe_muons["mass"]

    n_mother_muon = ak.zip({
        "pt":   lhe_muons_pt[:, 0:1],
        "eta":  lhe_muons_eta[:, 0:1],
        "phi":  lhe_muons_phi[:, 0:1],
        "mass": lhe_muons_mass[:, 0:1]
    }, with_name="Momentum4D")

    wr_mother_muon = ak.zip({
        "pt":   lhe_muons_pt[:, 1:2],
        "eta":  lhe_muons_eta[:, 1:2],
        "phi":  lhe_muons_phi[:, 1:2],
        "mass": lhe_muons_mass[:, 1:2]
    }, with_name="Momentum4D")

    # Gen-level tops
    gen_pids   = events["GenPart_pdgId"].array()
    gen_pts    = events["GenPart_pt"].array()
    gen_etas   = events["GenPart_eta"].array()
    gen_phis   = events["GenPart_phi"].array()
    gen_masses = events["GenPart_mass"].array()
    top_mask   = (gen_pids == 6) | (gen_pids == -6)

    genparticle = ak.zip({
        "pt":   gen_pts,
        "eta":  gen_etas,
        "phi":  gen_phis,
        "mass": gen_masses
    }, with_name="Momentum4D")
    gentops     = genparticle[top_mask]

    gentop = ak.zip({
        "pt":   gentops["pt"][:, 0:1],
        "eta":  gentops["eta"][:, 0:1],
        "phi":  gentops["phi"][:, 0:1],
        "mass": gentops["mass"][:, 0:1]
    }, with_name="Momentum4D")

    # Reco-level objects
    ak4 = ak.zip({
        "pt":   events["Jet_pt"].array(),
        "eta":  events["Jet_eta"].array(),
        "phi":  events["Jet_phi"].array(),
        "mass": events["Jet_mass"].array()
    }, with_name="Momentum4D")

    fatjets = ak.zip({
        "pt":   events["FatJet_pt"].array(),
        "eta":  events["FatJet_eta"].array(),
        "phi":  events["FatJet_phi"].array(),
        "mass": events["FatJet_mass"].array()
    }, with_name="Momentum4D")

    reco_muons = ak.zip({
        "pt":   events["Muon_pt"].array(),
        "eta":  events["Muon_eta"].array(),
        "phi":  events["Muon_phi"].array(),
        "mass": events["Muon_mass"].array()
    }, with_name="Momentum4D")
    
    btagging = events["FatJet_btagDeepB"].array()
    toptagging = events["FatJet_particleNetWithMass_TvsQCD"].array()
    softdrop_mass = events["FatJet_msoftdrop"].array()
    first_muon = reco_muons[:, 0:1]  # 첫 번째 뮤온
    second_muon = reco_muons[:, 1:2]  # 두 번째 뮤온
    sortidx = ak.argsort(-reco_muons["pt"], axis=1)  # pt 기준으로 내림차순 정렬
    first_muon = reco_muons[sortidx][:, 0:1]  # 가장 큰 pt를 가진 뮤온
    second_muon = reco_muons[sortidx][:, 1:2]  # 두 번째로 큰 pt를 가진 뮤온
    
    # Overlap removal
    muon_iso = events["Muon_tkRelIso"].array() 
    muon_highpt_id = events["Muon_highPtId"].array()

    leading_muon_iso = muon_iso[sortidx][:, 0:1]
    leading_muon_hpt = muon_highpt_id[sortidx][:, 0:1]
    subleading_muon_iso = muon_iso[sortidx][:, 1:2]
    subleading_muon_hpt = muon_highpt_id[sortidx][:, 1:2]

    first_muon_cleaned  = overlap_removal(first_muon, [reco_muons], cut=0.4)
    second_muon_cleaned = overlap_removal(second_muon, [reco_muons], cut=0.4)

    
    # Apply event selection
    pt1 = ak.sum(first_muon_cleaned["pt"], axis=1)
    pt2 = ak.sum(second_muon_cleaned["pt"], axis=1)
    eta1= ak.sum(first_muon_cleaned["eta"], axis=1)
    eta2= ak.sum(second_muon_cleaned["eta"], axis=1)
    hlt = events["HLT_IsoMu30"].array()


    hlt_mask = (
        (pt1 > 30) & (pt2 > 0)  & (hlt==True)
        & (abs(eta1) < 2.4) & (abs(eta2) < 2.4)
    )
    lmi = leading_muon_iso[hlt_mask]
    lmhpt = leading_muon_hpt[hlt_mask]
    slmi = subleading_muon_iso[hlt_mask]
    slhpt = subleading_muon_hpt[hlt_mask]
    fm1 = first_muon_cleaned[hlt_mask]
    fm2 = second_muon_cleaned[hlt_mask]
    
    
    # Dilepton mass cut
    mll = (fm1 + fm2).mass
    mll_mask = mll > 0
    fm1 = fm1[mll_mask]
    fm2 = fm2[mll_mask]
    

    # Final vars
    
    
    mu1_pt_flat   = ak.flatten(fm1["pt"])
    mu2_pt_flat   = ak.flatten(fm2["pt"])
    mll_flat      = ak.flatten(mll)
    totalev       = len(lhe_pt)
    leftev        = len(mu1_pt_flat)
    hltpassev     = len(ak.flatten(first_muon_cleaned["pt"][hlt_mask]))


    return mu1_pt_flat, mu2_pt_flat, mll_flat ,lmi , slmi, lmhpt, slhpt ,totalev , leftev ,hltpassev 

def iter_allfile (path):
    from importlib import import_module
    import os
    import sys
    import argparse
    import linecache
    import uproot
    import vector
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import awkward as ak
    from tqdm import tqdm  # ✅ 진행률 표시
    import glob
    
    file_list = sorted(glob.glob(path + "*.root"))
    
    combined_p4_list = []
    mll_list = []
    lb_list = []
    lt_list = []
    mu1_pt_list = []
    mu2_pt_list = []
    lmi_list = []
    lmh_list = []
    smi_list = []
    smh_list = []
    btag_list = []
    totalev_list = 0
    leftev_list = 0
    hltpassev_list = 0

    # tqdm으로 파일 리스트 순회
    for sample in tqdm(file_list, desc="Processing ROOT files"):

        mu1_pt_flat, mu2_pt_flat, mll_flat ,lmi , slmi, lmhpt, slhpt ,totalev , leftev ,hltpassev = twoak8cleancut(sample)

        
        mll_list.append(mll_flat)
        mu1_pt_list.append(mu1_pt_flat)
        mu2_pt_list.append(mu2_pt_flat)
        totalev_list += totalev
        leftev_list += leftev
        hltpassev_list += hltpassev
        lmi_list.append(lmi)
        lmh_list.append(lmhpt)
        smi_list.append(slmi)
        smh_list.append(slhpt)
    return ( mll_list, mu1_pt_list, mu2_pt_list, lmi_list, lmh_list, smi_list, smh_list,totalev_list, leftev_list, hltpassev_list)
            #0: mll_list
            #1: mu1_pt_list
            #2: mu2_pt_list
            #3: lmi_list
            #4: lmh_list
            #5: smi_list
            #6: smh_list
            #7: totalev_list
            #8: leftev_list
            #9: hltpassev_list

def compute_expected_events(sample_info, iter_allfile, lumi_fb):
    """
    sample_info: dict, 최소 다음 키를 가져야 합니다.
      - "xsec": float    # cross section [pb]
      - "sumW": float    # genEventSumw
      - "path": str OR list[str]
           * str: 디렉터리 경로 (끝에 / 까지 포함)
           * list: 파일 경로 리스트

    iter_allfile: function(path_or_dir)
      # 원래 사용하시던 대로, str을 넘기면 그 경로(또는 디렉터리)에서
      # glob 혹은 자체 로직으로 Events를 읽어서 통과 이벤트 개수를 return

    lumi_fb: float, 분석 luminosity [fb^-1]

    return: dict {
      "Npass": int,    # iter_allfile 결과
      "w_evt": float,  # per-event weight
      "Nexp": float    # 기대 이벤트 수
    }
    """
    p = sample_info["path"]
    # 1) iter_allfile에 넘길 인자 결정
    if isinstance(p, list):
        # 리스트가 주어지면, 첫 번째 엔트리의 디렉터리만 뽑아서 넘김
        first = p[0]
        base_dir = os.path.dirname(first) + os.sep
        Npass = iter_allfile(base_dir)
        mll = Npass[0]
        ptmu1 = Npass[1]
        ptmu2 = Npass[2]
        tot = Npass[7]
        Npass = ak.flatten(ptmu1, axis=None)
        Npass = len(Npass)  # Npass는 통과 이벤트 개수

        
    elif isinstance(p, str):
        # 문자열이 디렉터리인지 파일인지 상관없이 그대로 넘김
        Npass = iter_allfile(p)
        mll = Npass[0]
        ptmu1 = Npass[1]
        ptmu2 = Npass[2]
        tot = Npass[7]
        Npass = ak.flatten(ptmu1, axis=None)
        Npass = len(Npass)  # Npass는 통과 이벤트 개수
    else:

        raise ValueError(f"Unexpected type for sample_info['path']: {type(p)}")

    # 2) per-event normalization weight 계산 (pb→fb 변환 위해 *1000)
    xsec_pb = sample_info["xsec"]
    sumW    = sample_info["sumW"]
    w_evt   = xsec_pb * lumi_fb / sumW

    # 3) 기대 이벤트 수
    Nexp = Npass * w_evt

    return {"Npass": Npass, "w_evt": w_evt, "Nexp": Nexp, "mll": mll, "ptmu1": ptmu1, "ptmu2": ptmu2}

lumi = 300
info = json.load(open("./QCDB_HT40to100.json"))
print(info)
# 4) 계산 실행
res = compute_expected_events(info, iter_allfile, lumi)
print(f"Passed events    : {res['Npass']}")
print(f"Per-event weight : {res['w_evt']}")
print(f"Expected yield   : {res['Nexp']} events at {lumi} fb^-1")


flat = ak.flatten(res["mll"], axis=None)   # 1차원으로 모두 펼치기
mll_list = flat.tolist()
plt.hist(mll_list, bins=100, range=(0, 8000), histtype='step', label='ll reco')
plt.xlabel('mll [GeV]')
plt.ylabel('Entries')
plt.legend()

out_filename = 'mll_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")

plt.figure()
flat = ak.flatten(res["ptmu1"], axis=None)   # 1차원으로 모두 펼치기
ptmu1_list = flat.tolist()
plt.hist(ptmu1_list, bins=100, range=(0, 8000), histtype='step', label='mu1 reco')
flat = ak.flatten(res["ptmu2"], axis=None)   # 1차원으로 모두 펼치기
ptmu2_list = flat.tolist()
plt.hist(ptmu2_list, bins=100, range=(0, 8000), histtype='step', label='mu2 reco')
plt.xlabel('pt [GeV]')
plt.ylabel('Entries')
plt.legend()

out_filename = 'ptmu_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")



import numpy as np
import pandas as pd



## ll mass 
flat = ak.flatten(res["mll"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0, 100, 200 ,300, 400, 500, 600, 700, 800, 900, 1000,1500,2000, np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "mass_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("mll_counts.csv", index=False)    

## mu1 pt
flat = ak.flatten(res["ptmu1"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0,50,100,150,200,250,450,650,850,1000,1500,2000,np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "pt_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("ptmu1_counts.csv", index=False)

## mu2 pt
flat = ak.flatten(res["ptmu2"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0,50,100,150,200,250,450,650,850,1000,1500,2000,np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "pt_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("ptmu2_counts.csv", index=False)
