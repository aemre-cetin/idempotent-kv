import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# --- USPTO Çizim Standartları Konfigürasyonu ---
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

# ==============================================================================
# SAYFA 1: FIG. 1 - Donanım ve Sistem Mimarisi Blok Şeması
# ==============================================================================
def create_fig1():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Dış Kutu: Hızlandırıcı (100)
    outer = patches.Rectangle((6, 12), 88, 78, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(outer)
    ax.text(9, 87.5, "100: NEURAL NETWORK ACCELERATOR (GPU / NPU)", fontweight='bold', fontsize=10.5)

    # Üst Blok: Birincil HBM Belleği (102)
    hbm = patches.Rectangle((10, 69), 80, 15, fill=False, edgecolor='black', linewidth=1.2)
    ax.add_patch(hbm)
    ax.text(12, 80.5, "102: PRIMARY ACCELERATOR MEMORY (HBM / VRAM)", fontweight='bold', fontsize=9.5)
    
    # 104: Bellek İçindeki Tensör Blokları
    ax.text(12, 76.5, "104: Contiguous Dynamic KV-Cache Tensor Buffer", fontsize=8.5, fontstyle='italic')
    for i in range(6):
        x = 12 + i * 12.8
        blk = patches.Rectangle((x, 70.5), 11.5, 4.8, fill=False, edgecolor='black', linewidth=0.9)
        ax.add_patch(blk)
        label = f"Block {i}" if i < 5 else "Block N-1"
        ax.text(x + 5.75, 72.9, label, ha='center', va='center', fontsize=8)

    # İki Yönlü Veri Yolu Oku (Memory Bus)
    ax.annotate('', xy=(50, 69), xytext=(50, 61),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(52, 64.5, "Direct High-Speed Memory Bus\n(Zero Intermediate OS Allocations)", fontsize=8)

    # Orta Blok: Bellek Yönetim Denetleyicisi (110)
    ctrl = patches.Rectangle((10, 27), 80, 34, fill=False, edgecolor='black', linewidth=1.2)
    ax.add_patch(ctrl)
    ax.text(12, 57.5, "110: INTEGRATED MEMORY MANAGEMENT CONTROLLER", fontweight='bold', fontsize=9.5)

    # 112: İlişkisel Skorlama ve Sanal CAM
    u112 = patches.Rectangle((14, 47.5), 72, 7.8, fill=False, edgecolor='black', linewidth=0.9)
    ax.add_patch(u112)
    ax.text(16, 52.5, "112: Associative Scoring & Characteristic Mapping Unit", fontweight='bold', fontsize=8.5)
    ax.text(16, 49.5, "- Tracks attention scores S(k_i); projects to virtual CAM tiers", fontsize=8)

    # 114: İdempotent Projeksiyon
    u114 = patches.Rectangle((14, 38), 72, 7.5, fill=False, edgecolor='black', linewidth=0.9)
    ax.add_patch(u114)
    ax.text(16, 43, "114: Idempotent Projection Engine", fontweight='bold', fontsize=8.5)
    ax.text(16, 40, "- Enforces f(f(x)) = f(x); guarantees single-pass convergence", fontsize=8)

    # 116: Yerinde Döngü Çekirdeği
    u116 = patches.Rectangle((14, 29), 72, 7.2, fill=False, edgecolor='black', linewidth=0.9)
    ax.add_patch(u116)
    ax.text(16, 33.8, "116: In-Place Disjoint Cycle-Leader Execution Kernel", fontweight='bold', fontsize=8.5)
    ax.text(16, 30.8, "- O(1) auxiliary registers; executes zero-copy cyclic permutations", fontsize=8)

    # Kontrol Biriminden Çekirdeklere Yönlendirme Oku
    ax.annotate('', xy=(50, 27), xytext=(50, 21),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Alt Blok: Akış Çoklu İşlemcileri (140)
    sm = patches.Rectangle((10, 14), 80, 7, fill=False, edgecolor='black', linewidth=1.2)
    ax.add_patch(sm)
    ax.text(50, 18.5, "140: STREAMING MULTIPROCESSORS / TENSOR CORES", fontweight='bold', ha='center', fontsize=9)
    ax.text(50, 15.7, "Parallel Attention Matrix Multiplication: Softmax(Q * K^T / sqrt(d)) * V", ha='center', fontsize=7.8)

    # USPTO Şekil Numarası (Altta Ortalanmış)
    ax.text(50, 5, "FIG. 1", fontweight='bold', fontsize=12, ha='center')
    return fig

# ==============================================================================
# SAYFA 2: FIG. 2 - Bellek Durum Dönüşümü Şeması
# ==============================================================================
def create_fig2():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # --- 1. Bölüm: Sıkıştırma Öncesi (Before Compaction) ---
    ax.text(8, 91, "STATE A: BEFORE COMPACTION (FRAGMENTED BUFFER)", fontweight='bold', fontsize=10)
    tokens_before = [("T_0", "High"), ("T_1", "Low"), ("T_2", "High"),
                     ("T_3", "Low"),  ("T_4", "High"), ("T_5", "Low")]
    
    for i, (tok, util) in enumerate(tokens_before):
        x = 8 + i * 14.5
        box = patches.Rectangle((x, 79), 13.5, 9.5, fill=False, edgecolor='black', linewidth=1.1)
        ax.add_patch(box)
        ax.text(x + 6.75, 84.8, f"Index [{i}]\n{tok}", ha='center', va='center', fontweight='bold', fontsize=8.5)
        ax.text(x + 6.75, 75.5, f"Utility: {util}", ha='center', fontsize=8)

    # --- 2. Bölüm: İdempotent Haritalama & Döngü Ayrışımı ---
    ax.text(8, 68, "ASSOCIATIVE TARGET MAP f(x) & DISJOINT CYCLE RESOLUTION", fontweight='bold', fontsize=10)
    map_box = patches.Rectangle((8, 50), 84, 15, fill=False, edgecolor='black', linewidth=1.0)
    ax.add_patch(map_box)
    
    ax.text(11, 61.5, "f(0) = 0   [Fixed Point / Stabilized Attractor Basin - No Movement]", fontsize=8.8, fontweight='bold')
    ax.text(11, 57, "f(1)=3,  f(2)=1,  f(3)=5,  f(4)=2,  f(5)=4", fontsize=8.8)
    ax.text(11, 52.5, "Identified Disjoint Cycle: ( 1  ->  3  ->  5  ->  4  ->  2  ->  1 )", fontsize=8.8, fontweight='bold')

    # Temp Register Kutucuğu
    temp_box = patches.Rectangle((62, 35), 30, 9.5, fill=False, edgecolor='black', linewidth=1.1)
    ax.add_patch(temp_box)
    ax.text(77, 41, "Temp_Reg [O(1) Memory]", fontweight='bold', ha='center', fontsize=8.5)
    ax.text(77, 37.5, "Holds Payload of Index 1", ha='center', fontsize=8)

    # Takas Yönergeleri
    ax.text(8, 44, "IN-PLACE CYCLE EXECUTION STEPS:", fontweight='bold', fontsize=9)
    steps = [
        "1. Temp_Reg <-- Load(Block 1)",
        "2. Block 1 <-- Block 2   (f(2)=1)",
        "3. Block 2 <-- Block 4   (f(4)=2)",
        "4. Block 4 <-- Block 5   (f(5)=4)",
        "5. Block 5 <-- Block 3   (f(3)=5)",
        "6. Block 3 <-- Temp_Reg  (Cycle Closed)"
    ]
    for idx, s in enumerate(steps):
        ax.text(10, 40 - idx * 2.5, s, fontsize=8.2)

    # --- 3. Bölüm: Sıkıştırma Sonrası (After Compaction) ---
    ax.text(8, 22, "STATE B: AFTER COMPACTION (CONTIGUOUS ACTIVE BUFFER)", fontweight='bold', fontsize=10)
    tokens_after = [("T_0", "Active"), ("T_2", "Active"), ("T_4", "Active"),
                    ("T_1", "Evicted"), ("T_5", "Evicted"), ("T_3", "Evicted")]
    
    for i, (tok, status) in enumerate(tokens_after):
        x = 8 + i * 14.5
        box = patches.Rectangle((x, 10.5), 13.5, 9.5, fill=False, edgecolor='black', linewidth=1.1)
        ax.add_patch(box)
        ax.text(x + 6.75, 16.3, f"Index [{i}]\n{tok}", ha='center', va='center', fontweight='bold', fontsize=8.5)
        ax.text(x + 6.75, 7.5, status, ha='center', fontsize=8, fontstyle='italic')

    # Sınır Çizgisi (Truncation Boundary)
    ax.plot([50.8, 50.8], [6, 21.5], color='black', linestyle='--', linewidth=1.8)
    ax.annotate('Allocation Boundary Pointer\n(Buffer Truncated Here)', xy=(50.8, 21.5), xytext=(54, 25.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.1), fontsize=8.5, fontweight='bold')

    ax.text(28, 4.5, "[ Contiguous Active Region ]", ha='center', fontsize=8.5, fontweight='bold')
    ax.text(74, 4.5, "[ Released / Truncated Memory ]", ha='center', fontsize=8.5)

    # USPTO Şekil Numarası
    ax.text(50, 1.2, "FIG. 2", fontweight='bold', fontsize=12, ha='center')
    return fig

# ==============================================================================
# SAYFA 3: FIG. 3 - Algoritma Mantıksal Akış Şeması (Flowchart)
# ==============================================================================
def create_fig3():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, text, ref_num=""):
        b = patches.Rectangle((x - w/2, y - h/2), w, h, fill=False, edgecolor='black', linewidth=1.1)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
        if ref_num:
            ax.text(x - w/2 - 2, y + h/2 - 0.5, ref_num, fontweight='bold', fontsize=8.5, ha='right')

    def draw_diamond(x, y, r, text, ref_num=""):
        pts = [[x, y + r], [x + r*1.6, y], [x, y - r], [x - r*1.6, y]]
        poly = patches.Polygon(pts, fill=False, edgecolor='black', linewidth=1.1)
        ax.add_patch(poly)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.8)
        if ref_num:
            ax.text(x - r*1.6 - 2, y + r - 0.5, ref_num, fontweight='bold', fontsize=8.5, ha='right')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
        if label:
            ax.text((x1 + x2)/2 + 2, (y1 + y2)/2, label, fontsize=8, fontweight='bold')

    # Akış Adımları
    draw_box(50, 94, 38, 4.5, "START: Dynamic Compaction Triggered", "302")
    draw_arrow(50, 91.75, 50, 87.5)

    draw_box(50, 85, 52, 5, "Generate Associative Target Vector f[0..N-1]\nEnforcing Idempotent Condition: f(f(x)) = f(x)", "304")
    draw_arrow(50, 82.5, 50, 78.5)

    draw_box(50, 76.5, 30, 4, "Initialize Index: i = 0", "306")
    draw_arrow(50, 74.5, 50, 70)

    draw_diamond(50, 66.5, 3.5, "Is i >= N ?", "308")
    draw_arrow(50, 63, 50, 56.5, "NO")
    
    # YES oku (Bitiş)
    draw_arrow(55.6, 66.5, 80, 66.5, "YES")
    draw_box(80, 60, 26, 6, "310: Truncate Active\nBuffer Boundary Pointer\n& Terminate", "")

    # Sabit Nokta Kontrolü
    draw_diamond(50, 53, 3.5, "Is f[i] == i ?", "312")
    draw_arrow(50, 49.5, 50, 44, "NO")
    
    # Sabit nokta ise atla
    ax.plot([55.6, 75, 75], [53, 53, 22], color='black', lw=1.1)
    ax.annotate('', xy=(59, 22), xytext=(75, 22), arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
    ax.text(58, 54, "YES (Fixed Point)", fontsize=7.8)

    # 314: Döngü Arama
    draw_box(50, 41.5, 46, 4.5, "Initialize: curr = f[i], is_leader = True", "314")
    draw_arrow(50, 39.25, 50, 34.5)

    # 316: curr < i Kontrolü
    draw_diamond(50, 31, 3.5, "Is curr < i ?", "316")
    draw_arrow(50, 27.5, 50, 23, "NO")
    
    # curr < i YES ise atla
    ax.plot([44.4, 25, 25], [31, 31, 22], color='black', lw=1.1)
    ax.annotate('', xy=(41, 22), xytext=(25, 22), arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
    ax.text(28, 32, "YES (Already Visited)", fontsize=7.5)

    # 320: Döngüyü İlerlet
    draw_box(50, 21, 36, 4.2, "curr = f[curr]\nRepeat until curr == i (Cycle Closed)", "320")
    draw_arrow(50, 18.9, 50, 14.5)

    # 324 & 326: Yerinde Takas
    draw_box(50, 11.5, 56, 5.5, "324: Load Block[i] into Temp_Reg (O(1) Space)\n326: Shift Blocks along Cycle Orbit; Close Loop with Temp_Reg", "")
    draw_arrow(50, 8.75, 50, 6)

    # İndis Artırma
    draw_box(50, 4.2, 30, 3.5, "Increment: i = i + 1", "328")
    
    # Geriye Dönüş Hattı (308'e döner)
    ax.plot([35, 10, 10, 44], [4.2, 4.2, 70, 70], color='black', lw=1.1)
    ax.annotate('', xy=(44.4, 70), xytext=(38, 70), arrowprops=dict(arrowstyle='->', color='black', lw=1.1))

    # USPTO Şekil Numarası
    ax.text(50, 0.8, "FIG. 3", fontweight='bold', fontsize=12, ha='center')
    return fig

# ==============================================================================
# ÇOK SAYFALI RESMİ USPTO PDF DIŞA AKTARIMI
# ==============================================================================
def generate_all_drawings():
    output_pdf = "Drawings_KV_Cache_Compaction.pdf"
    
    with PdfPages(output_pdf) as pdf:
        # Sayfa 1
        f1 = create_fig1()
        pdf.savefig(f1, bbox_inches='tight')
        plt.close(f1)
        
        # Sayfa 2
        f2 = create_fig2()
        pdf.savefig(f2, bbox_inches='tight')
        plt.close(f2)
        
        # Sayfa 3
        f3 = create_fig3()
        pdf.savefig(f3, bbox_inches='tight')
        plt.close(f3)

    print(f">> Başarılı! 3 sayfalık resmi patent çizim dosyası oluşturuldu: {output_pdf}")

if __name__ == "__main__":
    generate_all_drawings()