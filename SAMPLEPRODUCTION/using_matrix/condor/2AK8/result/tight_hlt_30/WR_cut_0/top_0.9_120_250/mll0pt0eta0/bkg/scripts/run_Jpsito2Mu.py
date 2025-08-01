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

    signaltop_cleaned = overlap_removal(fatjets, [first_muon_cleaned,second_muon_cleaned], cut=0.8)
    toptaggedak8 = toptagging_overlap_removal(fatjets, [first_muon_cleaned,second_muon_cleaned], cut=0.8,events=events)
    softdrop_mass_ak8 = sdm_toptagging_overlap_removal(fatjets, [first_muon_cleaned,second_muon_cleaned], cut=0.8,events=events)
    #btaggingak8 = btagging_overlap_removal(fatjets, [first_muon_cleaned,second_muon_cleaned], cut=0.8)

    idx_desc = ak.argsort(-signaltop_cleaned["pt"], axis=1)

    toptagging_mask = toptaggedak8 > 0.9  # ParticleNet T vs QCD threshold
    softdrop_mass_mask = (softdrop_mass_ak8 > 120) & (softdrop_mass_ak8 < 250)   # SoftDrop mass threshold
    total_top_mask = toptagging_mask & softdrop_mass_mask

    cleaned_toptagged_ak8 = signaltop_cleaned[total_top_mask]  # top tagging 된 애들만 남김
    idx_cleaned_toptagged_ak8 = ak.argsort(-cleaned_toptagged_ak8["pt"], axis=1)
    leading_toptagged_ak8 = cleaned_toptagged_ak8[idx_cleaned_toptagged_ak8][:, 0:1]  # 가장 큰 pt

    ## top을 제외한 나머지 ak8 중 제일 큰 pt

    cleaned_bjet_ak8 = signaltop_cleaned[~toptagging_mask]
    idx_cleaned_bjet_ak8 = ak.argsort(-cleaned_bjet_ak8["pt"], axis=1)
    leading_bjet_ak8 = cleaned_bjet_ak8[idx_cleaned_bjet_ak8][:, 0:1]  # 가장 큰 pt

    # Apply event selection
    pt1 = ak.sum(first_muon_cleaned["pt"], axis=1)
    pt2 = ak.sum(second_muon_cleaned["pt"], axis=1)
    eta1= ak.sum(first_muon_cleaned["eta"], axis=1)
    eta2= ak.sum(second_muon_cleaned["eta"], axis=1)
    pt_leading_topjets = ak.sum(leading_toptagged_ak8["pt"], axis=1)
    pt_leading_bjets   = ak.sum(leading_bjet_ak8["pt"], axis=1)
    hlt = events["HLT_IsoMu30"].array()
    
    mask_evt = (
        (pt1 > 0) & (pt2 > 0) &
        #(abs(eta1) < 2.4) & (abs(eta2) < 2.4) &
        (pt_leading_topjets > 0) & (pt_leading_bjets > 0)
    )
    
    hlt_mask = (
        (pt1 > 0) & (pt2 > 0) &
        (pt_leading_topjets > 0) & (pt_leading_bjets > 0) & (hlt==True)
    )
    hltpassev     = len(ak.flatten(first_muon_cleaned["pt"][hlt_mask]))
    first_muon_cleaned  = first_muon_cleaned[mask_evt]
    second_muon_cleaned = second_muon_cleaned[mask_evt]
    leading_topjets     = leading_toptagged_ak8[mask_evt]
    leading_bjets       = leading_bjet_ak8[mask_evt]
    
    lmi = leading_muon_iso[mask_evt]
    lmhpt = leading_muon_hpt[mask_evt]
    slmi = subleading_muon_iso[mask_evt]
    slhpt = subleading_muon_hpt[mask_evt]
    # Dilepton mass cut
    mll = (first_muon_cleaned + second_muon_cleaned).mass
    mll_mask = mll > 0
    first_muon_cleaned  = first_muon_cleaned[mll_mask]
    second_muon_cleaned = second_muon_cleaned[mll_mask]
    leading_topjets     = leading_topjets[mll_mask]
    leading_bjets       = leading_bjets[mll_mask]
    
    # Final variables
    combined_p4               = (first_muon_cleaned + second_muon_cleaned + leading_topjets + leading_bjets).mass
    mN                        = (first_muon_cleaned + leading_topjets + leading_bjets).mass
    first_muon_cleaned_pt_sum = ak.sum(first_muon_cleaned["pt"], axis=1)
    second_muon_cleaned_pt_sum= ak.sum(second_muon_cleaned["pt"], axis=1)
    mll                       = (first_muon_cleaned + second_muon_cleaned).mass
    total_events = len(lhe_pt)
    left_events = len(combined_p4)
    
    
    return combined_p4, mN, first_muon_cleaned_pt_sum, second_muon_cleaned_pt_sum, mll,leading_topjets["pt"],leading_bjets["pt"] ,lmi,slmi, lmhpt, slhpt ,total_events, left_events,hltpassev

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

        combined_p4, mN, mu1_pt, mu2_pt, mll,lt,lb ,lmi,slmi, lmhpt, slhpt ,totalev, leftev,hltpassev = twoak8cleancut(sample)

        combined_p4_list.append(combined_p4)
        mll_list.append(mll)
        mu1_pt_list.append(mu1_pt)
        mu2_pt_list.append(mu2_pt)
        lt_list.append(lt)
        lb_list.append(lb)
        totalev_list += totalev
        leftev_list += leftev
        hltpassev_list += hltpassev
        lmi_list.append(lmi)
        lmh_list.append(lmhpt)
        smi_list.append(slmi)
        smh_list.append(slhpt)
    return (combined_p4_list, mll_list, mu1_pt_list, mu2_pt_list,lt_list,lb_list, lmi_list, lmh_list, smi_list, smh_list,totalev_list, leftev_list, hltpassev_list)
            # 0. combined_p4_list
            # 1. mll_list
            # 2. mu1_pt_list
            # 3. mu2_pt_list
            # 4. lt_list
            # 5. lb_list
            # 6. lmi_list
            # 7. lmh_list
            # 8. smi_list
            # 9. smh_list
            # 10. totalev_list
            # 11. leftev_list
            # 12. hltpassev_list


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
        mwr = Npass[0]
        mll = Npass[1]
        ptmu1 = Npass[2]
        ptmu2 = Npass[3]
        ptt = Npass[4]
        ptb = Npass[5]
        tot = Npass[10]
        Npass = (Npass[12])
        
    elif isinstance(p, str):
        # 문자열이 디렉터리인지 파일인지 상관없이 그대로 넘김
        Npass = iter_allfile(p)
        mwr = Npass[0]
        mll = Npass[1]
        ptmu1 = Npass[2]
        ptmu2 = Npass[3]
        ptt = Npass[4]
        ptb = Npass[5]
        tot = Npass[10]
        Npass = (Npass[12])
    else:

        raise ValueError(f"Unexpected type for sample_info['path']: {type(p)}")

    # 2) per-event normalization weight 계산 (pb→fb 변환 위해 *1000)
    xsec_pb = sample_info["xsec"]
    sumW    = sample_info["sumW"]
    w_evt   = xsec_pb * 1000 * lumi_fb / sumW

    # 3) 기대 이벤트 수
    Nexp = Npass * w_evt

    return {"Npass": Npass, "w_evt": w_evt, "Nexp": Nexp,"WRmass": mwr, "mll": mll, "ptmu1": ptmu1, "ptmu2": ptmu2, "ptt": ptt, "ptb": ptb}


lumi = 300
info = json.load(open("./Jpsito2Mu.json"))
print(info)
# 4) 계산 실행
res = compute_expected_events(info, iter_allfile, lumi)
print(f"Passed events    : {res['Npass']}")
print(f"Per-event weight : {res['w_evt']}")
print(f"Expected yield   : {res['Nexp']} events at {lumi} fb^-1")

fig, ax = plt.subplots()
ax.hist(ak.flatten(res["WRmass"]), bins=100, range=(0, 8000),
        histtype='step', label='WR mass')

# 2) 레이블, 범례
ax.set_xlabel('WR mass [GeV]')
ax.set_ylabel('Entries')
ax.legend()

# 3) 우측 상단에 작은 텍스트 추가
textstr = (
    f"Passed events    : {res['Npass']}\n"
    f"Per-event weight : {res['w_evt']:.3f}\n"
    f"Expected yield   : {res['Nexp']:.1f} events\n"
    f"at {lumi:.0f} fb⁻¹"
)
# props for the textbox: no frame, small font
props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5)

ax.text(
    0.98, 0.98, textstr,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=props
)


# 2) 파일로 저장
out_filename = 'WRmass_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")

plt.figure()
plt.hist(ak.flatten(res["mll"]), bins=100, range=(0, 8000), histtype='step', label='ll reco')
plt.xlabel('mll [GeV]')
plt.ylabel('Entries')
plt.legend()

out_filename = 'mll_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")

plt.figure()
plt.hist(ak.flatten(res["ptmu1"]), bins=100, range=(0, 8000), histtype='step', label='mu1 reco')
plt.hist(ak.flatten(res["ptmu2"]), bins=100, range=(0, 8000), histtype='step', label='mu2 reco')
plt.xlabel('pt [GeV]')
plt.ylabel('Entries')
plt.legend()

out_filename = 'ptmu_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")

plt.figure()
plt.hist(ak.flatten(res["ptt"]), bins=100, range=(0, 8000), histtype='step', label='t reco')
plt.hist(ak.flatten(res["ptb"]), bins=100, range=(0, 8000), histtype='step', label='b reco')
plt.xlabel('pt [GeV]')
plt.ylabel('Entries')
plt.legend()
out_filename = 'ptt_ptb_histogram.png'
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram to {out_filename}")



import numpy as np
import pandas as pd

## wr mass 
flat = ak.flatten(res["WRmass"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0, 800,1000, 1200, 1600, 2000, 2500, 3000, 4000,5000, np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "mass_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("wrmass_counts.csv", index=False)

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

## t pt
flat = ak.flatten(res["ptt"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0,50,100,150,200,250,300,350,550,750,950,1150,1350,1550,np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "pt_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("ptt_counts.csv", index=False)

## b pt
flat = ak.flatten(res["ptb"], axis=None)   # 1차원으로 모두 펼치기
values = np.array(flat)
bins = [0,50,100,150,200,250,300,350,550,750,950,1150,1350,1550,np.inf]
counts, _ = np.histogram(values, bins=bins)
counts = counts* res["w_evt"]  # 가중치 적용
bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
# DataFrame으로 만들기
df = pd.DataFrame({
    "pt_bin": bin_labels,
    "count": counts
})
# CSV로 저장 (헤더 포함, 인덱스는 제외)
df.to_csv("ptb_counts.csv", index=False)

