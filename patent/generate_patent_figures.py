import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

def create_fig1(fig_num_str="FIG. 1"):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # 100: Outer Accelerator Box
    outer = FancyBboxPatch((6, 7), 88, 88, boxstyle="round,pad=0.5,rounding_size=1.5",
                           facecolor='white', edgecolor='black', linewidth=1.6)
    ax.add_patch(outer)
    ax.text(9, 91.5, "100: NEURAL NETWORK ACCELERATOR (GPU / NPU ARCHITECTURE)", fontweight='bold', fontsize=9.8)

    # 102: Primary Memory (HBM3e)
    hbm = FancyBboxPatch((9, 71), 82, 18, boxstyle="round,pad=0.3,rounding_size=1.0",
                         facecolor='#f9f9f9', edgecolor='black', linewidth=1.3)
    ax.add_patch(hbm)
    ax.text(12, 85.5, "102: PRIMARY ACCELERATOR MEMORY (HBM3e / VRAM)", fontweight='bold', fontsize=9.2)
    ax.text(12, 82.2, "104: Contiguous Dynamic KV-Cache Tensor Buffer [B, H, L, D_k]", 
            fontsize=8.0, fontstyle='italic')

    # KV Blocks inside HBM
    kv_labels = [
        ("Slot 0\n[K0, V0]", "Active"),
        ("Slot 1\n[K1, V1]", "Evict"),
        ("Slot 2\n[K2, V2]", "Active"),
        ("Slot 3\n[K3, V3]", "Evict"),
        ("Slot 4\n[K4, V4]", "Active"),
        ("Slot N-1\n[Kn, Vn]", "Tail")
    ]
    for i, (lbl, role) in enumerate(kv_labels):
        x = 11.5 + i * 13.0
        style = 'solid' if role == "Active" else ('dashed' if role == "Evict" else 'dashdot')
        fill_col = '#e8e8e8' if role == "Active" else '#ffffff'
        blk = FancyBboxPatch((x, 73.5), 11.5, 7.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor=fill_col, edgecolor='black', linestyle=style, linewidth=1.1)
        ax.add_patch(blk)
        ax.text(x + 5.75, 77.5, lbl, ha='center', va='center', fontsize=7.2, fontweight='bold')
        ax.text(x + 5.75, 74.5, f"({role})", ha='center', va='center', fontsize=6.5, fontstyle='italic')

    # Direct Memory Bus Callout
    ax.annotate('', xy=(50, 71), xytext=(50, 62.5), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(50, 67.2, "Direct High-Speed Memory Bus (Zero Intermediate cudaMalloc Calls)", 
            ha='center', fontsize=8.0, fontweight='bold')
    ax.text(50, 64.5, "Bandwidth: up to 8.0 TB/s (NVIDIA Blackwell sm_120 / HBM3e)", 
            ha='center', fontsize=7.5, fontstyle='italic')

    # 110: Integrated Memory Management Controller
    ctrl = FancyBboxPatch((9, 23), 82, 39, boxstyle="round,pad=0.4,rounding_size=1.2",
                          facecolor='#fcfcfc', edgecolor='black', linewidth=1.3)
    ax.add_patch(ctrl)
    ax.text(12, 58.5, "110: INTEGRATED IN-PLACE MEMORY MANAGEMENT CONTROLLER", fontweight='bold', fontsize=9.2)

    # 112 Unit
    u1 = FancyBboxPatch((12, 48.5), 76, 8.2, boxstyle="round,pad=0.3,rounding_size=0.7",
                        facecolor='#ffffff', edgecolor='black', linewidth=1.0)
    ax.add_patch(u1)
    ax.text(14, 53.8, "112: Associative Scoring & Characteristic Mapping Unit", fontweight='bold', fontsize=8.3)
    ax.text(14, 50.5, "- Tracks cumulative attention scores S(k_i); partitions active vs evicted token slots", fontsize=7.6)

    # 114 Unit
    u2 = FancyBboxPatch((12, 37.0), 76, 8.2, boxstyle="round,pad=0.3,rounding_size=0.7",
                        facecolor='#ffffff', edgecolor='black', linewidth=1.0)
    ax.add_patch(u2)
    ax.text(14, 42.3, "114: Idempotent Projection Engine", fontweight='bold', fontsize=8.3)
    ax.text(14, 39.0, "- Enforces f(f(x)) = f(x); locks active tokens into stabilized attractor basins", fontsize=7.6)

    # 116 Unit
    u3 = FancyBboxPatch((12, 25.5), 76, 8.2, boxstyle="round,pad=0.3,rounding_size=0.7",
                        facecolor='#ffffff', edgecolor='black', linewidth=1.0)
    ax.add_patch(u3)
    ax.text(14, 30.8, "116: In-Place Disjoint Cycle-Leader Permutation Engine", fontweight='bold', fontsize=8.3)
    ax.text(14, 27.5, "- O(1) auxiliary registers; executes zero-copy cyclic orbit swaps without bitmasks", fontsize=7.6)

    # Internal pipeline flow arrows
    ax.annotate('', xy=(50, 48.5), xytext=(50, 45.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(50, 37.0), xytext=(50, 33.9), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Arrow from 110 to 140
    ax.annotate('', xy=(50, 23), xytext=(50, 18), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 140: Tensor Cores
    tc = FancyBboxPatch((9, 9.5), 82, 8.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                        facecolor='#f0f0f0', edgecolor='black', linewidth=1.3)
    ax.add_patch(tc)
    ax.text(50, 14.8, "140: STREAMING MULTIPROCESSORS / TENSOR CORES (PARALLEL COMPUTE)", 
            fontweight='bold', ha='center', fontsize=8.5)
    ax.text(50, 11.8, "Vectorized Attention Matrix: Softmax(Q * K^T / sqrt(d)) * V on Compact Contiguous Memory", 
            ha='center', fontsize=7.4)

    if fig_num_str:
        ax.text(50, 3.0, fig_num_str, fontweight='bold', fontsize=12, ha='center')
    plt.tight_layout()
    return fig


def create_fig2(fig_num_str="FIG. 2"):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 97.0, "DYNAMIC KEY-VALUE (KV) CACHE IN-SITU STATE TRANSFORMATION", 
            ha='center', fontweight='bold', fontsize=10.2)

    # ==================== STATE A ====================
    ax.text(8, 92.5, "STATE A: FRAGMENTED KV-CACHE BUFFER (PRIOR TO COMPACTION)", fontweight='bold', fontsize=8.8)
    s_a_box = FancyBboxPatch((8, 80.0), 84, 11.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                             facecolor='#fafafa', edgecolor='black', linewidth=1.2)
    ax.add_patch(s_a_box)

    slots_A = [
        ("Slot [0]", "T[0]", "High Utility", True),
        ("Slot [1]", "T[1]", "Low Utility", False),
        ("Slot [2]", "T[2]", "High Utility", True),
        ("Slot [3]", "T[3]", "Low Utility", False),
        ("Slot [4]", "T[4]", "High Utility", True),
        ("Slot [5]", "T[5]", "Low Utility", False)
    ]
    for i, (s_idx, tok, util, active) in enumerate(slots_A):
        x = 9.5 + i * 13.5
        style = 'solid' if active else 'dashed'
        fill_col = '#e6e6e6' if active else '#ffffff'
        blk = FancyBboxPatch((x, 81.5), 12, 8.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor=fill_col, edgecolor='black', linestyle=style, linewidth=1.1)
        ax.add_patch(blk)
        ax.text(x + 6, 87.5, s_idx, ha='center', fontsize=7.2, fontweight='bold')
        ax.text(x + 6, 84.8, tok, ha='center', fontsize=8.2, fontweight='bold')
        ax.text(x + 6, 82.7, util, ha='center', fontsize=6.8, fontstyle='italic')

    # ==================== CYCLE RESOLUTION ====================
    ax.text(8, 75.5, "ASSOCIATIVE TARGET MAP f(x) & IN-PLACE DISJOINT CYCLE TRAVERSAL", fontweight='bold', fontsize=8.8)
    map_box = FancyBboxPatch((8, 36.0), 84, 38.0, boxstyle="round,pad=0.4,rounding_size=1.0",
                             facecolor='#ffffff', edgecolor='black', linewidth=1.2)
    ax.add_patch(map_box)

    # Top Formula Zone
    ax.text(11, 70.8, "Idempotent Invariant: f(f(x)) = f(x)   [Attractor Basin Locking]", fontweight='bold', fontsize=8.5)
    ax.text(11, 68.2, "* Fixed Point: f(0) = 0   [Retained at Origin - Zero Data Movement]", fontsize=7.8)
    ax.text(11, 65.6, "* Permutation Mapping: f(1)=3,  f(2)=1,  f(3)=5,  f(4)=2,  f(5)=4", fontsize=7.8)
    ax.text(11, 63.0, "* Disjoint Orbit: ( 1  -->  3  -->  5  -->  4  -->  2  -->  1 )", fontweight='bold', fontsize=8.3)

    # Temp Reg Box
    t_reg = FancyBboxPatch((13, 52.0), 22, 7.5, boxstyle="round,pad=0.3,rounding_size=0.6",
                           facecolor='#ececec', edgecolor='black', linewidth=1.2)
    ax.add_patch(t_reg)
    ax.text(24, 56.5, "Temp_Reg [O(1) Space]", ha='center', fontweight='bold', fontsize=7.6)
    ax.text(24, 53.5, "Holds Payload T[1]", ha='center', fontsize=7.0, fontstyle='italic')

    # Disjoint Cycle Blocks
    orbit_indices = [1, 2, 3, 4, 5]
    blk_xs = {}
    for idx_pos, slot_num in enumerate(orbit_indices):
        bx = 13 + idx_pos * 14.8
        blk_xs[slot_num] = bx
        b_patch = FancyBboxPatch((bx, 40.5), 12.0, 7.2, boxstyle="round,pad=0.2,rounding_size=0.4",
                                 facecolor='#f4f4f4', edgecolor='black', linewidth=1.0)
        ax.add_patch(b_patch)
        ax.text(bx + 6.0, 45.2, f"Slot [{slot_num}]", ha='center', fontsize=7.2, fontweight='bold')
        ax.text(bx + 6.0, 42.2, f"T[{slot_num}]", ha='center', fontsize=7.8)

    # 1. Load Slot 1 to Temp_Reg
    arrow_t1 = FancyArrowPatch((blk_xs[1] + 6.0, 47.7), (blk_xs[1] + 6.0, 52.0),
                               arrowstyle='->', color='black', lw=1.2, linestyle='dashed')
    ax.add_patch(arrow_t1)
    ax.text(blk_xs[1] + 6.0, 49.8, "1. Load", ha='right', fontsize=6.8, fontstyle='italic')

    # 2. Slot 2 -> Slot 1
    arrow_2_1 = FancyArrowPatch((blk_xs[2] + 3.0, 47.7), (blk_xs[1] + 9.0, 47.7),
                                connectionstyle="arc3,rad=0.35", arrowstyle='->', color='black', lw=1.2)
    ax.add_patch(arrow_2_1)
    ax.text((blk_xs[1] + blk_xs[2])/2 + 6.0, 51.5, "2. f(2)=1", ha='center', fontsize=6.6)

    # 3. Slot 4 -> Slot 2
    arrow_4_2 = FancyArrowPatch((blk_xs[4] + 3.0, 47.7), (blk_xs[2] + 9.0, 47.7),
                                connectionstyle="arc3,rad=0.35", arrowstyle='->', color='black', lw=1.2)
    ax.add_patch(arrow_4_2)
    ax.text((blk_xs[2] + blk_xs[4])/2 + 6.0, 55.0, "3. f(4)=2", ha='center', fontsize=6.6)

    # 4. Slot 5 -> Slot 4
    arrow_5_4 = FancyArrowPatch((blk_xs[5] + 3.0, 47.7), (blk_xs[4] + 9.0, 47.7),
                                connectionstyle="arc3,rad=0.35", arrowstyle='->', color='black', lw=1.2)
    ax.add_patch(arrow_5_4)
    ax.text((blk_xs[4] + blk_xs[5])/2 + 6.0, 51.5, "4. f(5)=4", ha='center', fontsize=6.6)

    # 5. Slot 3 -> Slot 5 (Curves underneath)
    arrow_3_5 = FancyArrowPatch((blk_xs[3] + 9.0, 40.5), (blk_xs[5] + 3.0, 40.5),
                                connectionstyle="arc3,rad=-0.35", arrowstyle='->', color='black', lw=1.2)
    ax.add_patch(arrow_3_5)
    ax.text((blk_xs[3] + blk_xs[5])/2 + 6.0, 37.0, "5. f(3)=5", ha='center', fontsize=6.6)

    # 6. Temp_Reg -> Slot 3 (Closes cycle)
    arrow_t_3 = FancyArrowPatch((35, 54.5), (blk_xs[3] + 6.0, 47.7),
                                connectionstyle="arc3,rad=0.18", arrowstyle='->',
                                color='black', lw=1.2, linestyle='dashed')
    ax.add_patch(arrow_t_3)
    ax.text(48.0, 52.5, "6. Close Cycle", ha='center', fontsize=6.8, fontstyle='italic')

    # ==================== STATE B ====================
    ax.text(8, 31.0, "STATE B: IN-SITU COMPACTED CONTIGUOUS BUFFER (POST-TRUNCATION)", fontweight='bold', fontsize=8.8)
    s_b_box = FancyBboxPatch((8, 12.0), 84, 17.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                             facecolor='#fafafa', edgecolor='black', linewidth=1.2)
    ax.add_patch(s_b_box)

    slots_B = [
        ("Slot [0]", "T[0]", "Active", True),
        ("Slot [1]", "T[2]", "Active (Swapped)", True),
        ("Slot [2]", "T[4]", "Active (Swapped)", True),
        ("Slot [3]", "T[1]", "Evicted (Moved)", False),
        ("Slot [4]", "T[5]", "Evicted (Moved)", False),
        ("Slot [5]", "T[3]", "Evicted (Moved)", False)
    ]
    for i, (s_idx, tok, status, active) in enumerate(slots_B):
        x = 9.5 + i * 13.5
        style = 'solid' if active else 'dotted'
        fill_col = '#e6e6e6' if active else '#ffffff'
        blk = FancyBboxPatch((x, 15.0), 12, 9.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor=fill_col, edgecolor='black', linestyle=style, linewidth=1.1)
        ax.add_patch(blk)
        ax.text(x + 6, 22.2, s_idx, ha='center', fontsize=7.2, fontweight='bold')
        ax.text(x + 6, 19.2, tok, ha='center', fontsize=8.2, fontweight='bold')
        ax.text(x + 6, 16.6, status, ha='center', fontsize=6.5, fontstyle='italic')

    # Truncation Boundary & Pointer
    ax.plot([50.0, 50.0], [12.5, 27.0], color='black', linestyle='--', linewidth=2.0)
    ax.annotate('Allocation Boundary Pointer\n(Buffer Truncated to K=3)', 
                xy=(50.0, 26.5), xytext=(55.0, 27.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=7.8, fontweight='bold')

    # Brackets under State B
    ax.annotate('', xy=(9.5, 13.2), xytext=(49.5, 13.2), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(29.5, 9.5, "Contiguous Active Working Context (K=3)\n[Direct Hardware Coalescing / FlashAttention]", 
            ha='center', fontsize=7.2, fontweight='bold')

    ax.annotate('', xy=(50.5, 13.2), xytext=(90.5, 13.2), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(70.5, 9.5, "Released / Truncated Memory Region\n[Overwritten on Subsequent Inferences]", 
            ha='center', fontsize=7.2)

    if fig_num_str:
        ax.text(50, 3.0, fig_num_str, fontweight='bold', fontsize=12, ha='center')
    plt.tight_layout()
    return fig


def create_fig3(fig_num_str="FIG. 3"):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, text, ref_num=""):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor='#ffffff', edgecolor='black', linewidth=1.1)
        ax.add_patch(p)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.8)
        if ref_num:
            ax.text(x - w/2 - 1.5, y + h/2 - 0.2, ref_num, fontweight='bold', fontsize=8.2, ha='right')

    def draw_diamond(x, y, rx, ry, text, ref_num=""):
        pts = [[x, y + ry], [x + rx, y], [x, y - ry], [x - rx, y]]
        poly = patches.Polygon(pts, facecolor='#fbfbfb', edgecolor='black', linewidth=1.1)
        ax.add_patch(poly)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.8)
        if ref_num:
            ax.text(x - rx - 1.5, y + ry, ref_num, fontweight='bold', fontsize=8.2, ha='right')

    # 302: Başlangıç
    draw_box(50, 95.0, 44, 4.2, "START: Dynamic KV-Cache Compaction Triggered", "302")
    ax.annotate('', xy=(50, 90.0), xytext=(50, 92.9), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 304: Hedef Harita
    draw_box(50, 86.8, 56, 5.5, "Generate Associative Target Index Vector f[0..N-1]\nEnforcing Idempotent Condition: f(f(x)) = f(x)", "304")
    ax.annotate('', xy=(50, 81.2), xytext=(50, 84.0), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 306: İndis Başlatma
    draw_box(50, 78.8, 30, 4.0, "Initialize Index: i = 0", "306")
    ax.annotate('', xy=(50, 73.8), xytext=(50, 76.8), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 308: Bitiş Kontrolü
    draw_diamond(50, 70.0, 11, 3.8, "Is i >= N ?", "308")
    ax.annotate('', xy=(50, 61.2), xytext=(50, 66.2), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.text(51.5, 64.0, "NO", fontsize=8, fontweight='bold')

    # 310: Bitiş Kutusu (YES)
    ax.annotate('', xy=(74, 70.0), xytext=(61, 70.0), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.text(64, 71.5, "YES", fontsize=8, fontweight='bold')
    draw_box(84.5, 70.0, 21, 6.0, "310: Truncate Active\nBuffer to K & Terminate", "")

    # 312: Sabit Nokta Kontrolü
    draw_diamond(50, 57.5, 12, 3.7, "Is f[i] == i ?", "312")
    ax.annotate('', xy=(50, 48.8), xytext=(50, 53.8), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.text(51.5, 51.5, "NO", fontsize=8, fontweight='bold')

    # 314: Arama Başlatma
    draw_box(50, 46.2, 46, 4.2, "Initialize: curr = f[i], is_leader = True", "314")
    ax.annotate('', xy=(50, 40.2), xytext=(50, 44.1), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 316: curr < i Kontrolü
    draw_diamond(50, 36.5, 12, 3.7, "Is curr < i ?", "316")
    ax.annotate('', xy=(50, 28.2), xytext=(50, 32.8), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.text(51.5, 30.5, "NO", fontsize=8, fontweight='bold')

    # 320: Döngü İlerleme
    draw_box(50, 25.5, 48, 4.8, "Advance Orbit: curr = f[curr]\nRepeat Traversal until curr == i (Cycle Closed)", "320")
    ax.annotate('', xy=(50, 19.5), xytext=(50, 23.1), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 324 & 326: Yerinde Takas Bloğu
    draw_box(50, 16.0, 62, 6.0, "324: Load Slot[i] Payload into Temp_Reg (O(1) Register Space)\n326: Shift Slots Along Cycle Orbit; Close Loop with Temp_Reg", "")
    ax.annotate('', xy=(50, 9.8), xytext=(50, 13.0), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 328: İndis Artırma
    draw_box(50, 7.5, 28, 3.8, "Increment: i = i + 1", "328")

    # 1. HAT: 312 YES (Fixed Point Skip)
    ax.plot([62, 88, 88, 64], [57.5, 57.5, 7.5, 7.5], color='black', lw=1.1)
    ax.annotate('', xy=(64, 7.5), xytext=(70, 7.5), arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
    ax.text(65, 59.0, "YES (Fixed Point - Skip)", fontsize=7.2)

    # 2. HAT: 316 YES (Not Cycle Leader Skip)
    ax.plot([62, 78, 78, 64], [36.5, 36.5, 8.5, 8.5], color='black', lw=1.1)
    ax.annotate('', xy=(64, 8.5), xytext=(70, 8.5), arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
    ax.text(64, 38.0, "YES (Not Leader - Skip)", fontsize=7.2)

    # 3. HAT: 328 -> 308 DÖNGÜSÜ
    ax.plot([36, 12, 12, 39], [7.5, 7.5, 70.0, 70.0], color='black', lw=1.2)
    ax.annotate('', xy=(39, 70.0), xytext=(30, 70.0), arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    if fig_num_str:
        ax.text(50, 2.5, fig_num_str, fontweight='bold', fontsize=12, ha='center')
    plt.tight_layout()
    return fig

def main():
    patent_dir = os.path.dirname(os.path.abspath(__file__))
    output_pdf = os.path.join(patent_dir, "Drawings_KV_Cache_Compaction.pdf")
    paper_fig_dir = os.path.join(os.path.dirname(patent_dir), "paper", "figures")
    pkg_fig_dir = os.path.join(os.path.dirname(os.path.dirname(patent_dir)), "packages", "idem-kv", "paper", "figures")
    pkg_pdf = os.path.join(os.path.dirname(os.path.dirname(patent_dir)), "patent-filing-packages", "01_KV_Cache_Compaction", "2_Drawings.pdf")
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(pkg_fig_dir, exist_ok=True)

    # A4 Dimensions: 210mm x 297mm = 8.2677 x 11.6929 inches (595.28 x 841.89 points)
    A4_W = 8.2677
    A4_H = 11.6929

    # 1. Patent drawings (with FIG. 1, FIG. 2, FIG. 3 labels for USPTO)
    with PdfPages(output_pdf) as pdf:
        f1 = create_fig1("FIG. 1")
        f1.set_size_inches(A4_W, A4_H)
        f1.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05)
        pdf.savefig(f1)
        plt.close(f1)

        f2 = create_fig2("FIG. 2")
        f2.set_size_inches(A4_W, A4_H)
        f2.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05)
        pdf.savefig(f2)
        plt.close(f2)

        f3 = create_fig3("FIG. 3")
        f3.set_size_inches(A4_W, A4_H)
        f3.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05)
        pdf.savefig(f3)
        plt.close(f3)

    # 2. Paper drawings (CLEAN without embedded "FIG. X" labels to avoid duplication with LaTeX captions)
    for target_dir in [paper_fig_dir, pkg_fig_dir]:
        f1_clean = create_fig1("")
        f1_clean.savefig(os.path.join(target_dir, "FIG_1_System_Architecture.pdf"), bbox_inches='tight')
        f1_clean.savefig(os.path.join(target_dir, "FIG_1_System_Architecture.png"), dpi=300, bbox_inches='tight')
        plt.close(f1_clean)

        f2_clean = create_fig2("")
        f2_clean.savefig(os.path.join(target_dir, "FIG_2_Memory_State_Transition.pdf"), bbox_inches='tight')
        f2_clean.savefig(os.path.join(target_dir, "FIG_2_Memory_State_Transition.png"), dpi=300, bbox_inches='tight')
        plt.close(f2_clean)

        f3_clean = create_fig3("")
        f3_clean.savefig(os.path.join(target_dir, "FIG_3_Algorithm_Flowchart.pdf"), bbox_inches='tight')
        f3_clean.savefig(os.path.join(target_dir, "FIG_3_Algorithm_Flowchart.png"), dpi=300, bbox_inches='tight')
        plt.close(f3_clean)

    print(f"[OK] New A4-Portrait high-res KV-Cache drawings generated at: {output_pdf}")
    print(f"[OK] Clean paper figures generated in: {paper_fig_dir} and {pkg_fig_dir}")

    if os.path.exists(os.path.dirname(pkg_pdf)):
        import shutil
        shutil.copyfile(output_pdf, pkg_pdf)
        print(f"[OK] Copied to Package: {pkg_pdf}")

if __name__ == "__main__":
    main()
