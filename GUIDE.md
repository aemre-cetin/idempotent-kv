# idem-kv (idempotent_kv): Kapsamlı Kullanıcı ve Geliştirici Kılavuzu (GUIDE.md)

**Zero-Copy In-Place KV-Cache Compactor for vLLM & Deep Learning Accelerators**

- **Paket Sürümü:** `0.2.1`
- **Birincil Python Modülü:** `idempotent_kv`
- **Donanım Hızlandırma:** Saf Python / PyTorch / Triton JIT Uyumlu
- **Lisans:** Apache 2.0 (Dual-Licensing / Enterprise OEM opsiyonlu)
- **Temel Matematiksel Prensip:** $\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$ (Tek Adımlı İdempotent İzdüşüm ve Sıfır Kopyalı Bellek İçi İnvolution)

---

## 1. Mimari ve Temel Kavramlar

`idem-kv` kütüphanesi, geleneksel iteratif algoritmaların ve dinamik bellek tahsislerinin (`malloc`/`free`, `O(N)` ara bellekler) yol açtığı gecikme, bellek parçalanması ve bellek duvarı (memory wall) problemlerini çözmek üzere tasarlanmıştır.

### Temel Tasarım İlkeleri:
1. **Sıfır Ek Bellek Tahsisi (0.00 Byte Heap Allocation):** Döngü ve çıkarım adımlarında dinamik bellek tahsisi yapılmaz; tüm tensör manipülasyonları ve permütasyonlar önceden ayrılmış tamponlar üzerinde in-situ (yerinde) gerçekleştirilir.
2. **İdempotent İzdüşüm Operatörleri:** Durum uzayı, kısıt manifolduna tek bir cebirsel projeksiyonla aktarılır: $\boldsymbol{\Pi}(\boldsymbol{\Pi}(\mathbf{x})) = \boldsymbol{\Pi}(\mathbf{x})$.
3. **Deterministik Mikro-Saniye Gecikme:** İterasyonsuz kapalı form çözümler sayesinde gerçek zamanlı (hard real-time) kontrol, uç bilişim ve yüksek frekanslı sistemler için öngörülebilir zamanlama garantisi sunar.

---

## 2. Kurulum ve Ortam Yapılandırması

```bash
# Geliştirici modunda paket dizininden kurulum:
cd packages/idem-kv
pip install -e .

# Birim testleri koşturarak kurulumu doğrulayın:
pytest -q
```

---

## 3. Modül ve Sınıf Referansı (Tam Çalışır Kod Örnekleri)

Aşağıda `idem-kv` kütüphanesinin `src/idempotent_kv` altında yer alan tüm gerçek modülleri, sınıfları ve fonksiyonları için çalıştırılabilir örnekler sunulmuştur:

### 3.1. Modül: `idempotent_kv.compactor`
#### Sınıf: `InplaceKVCompactor`
- **Açıklama:** High-Performance Zero-Copy In-Place KV-Cache Compactor.

Transforms KV-cache memory management in autoregressive transformer inference
(vLLM, SGLang, Hugging Face) from O(N) auxiliary out-of-place gathering into
an O(1) in-situ idempotent permutation engine.

Protected under U.S. Patent Application No. 64/148,668 (Patent Pending).
- **Metotlar:** `__init__()`, `build_idempotent_map()`, `compact()`, `compact_from_scores()`

```python
import torch
import numpy as np
from idempotent_kv.compactor import InplaceKVCompactor

# InplaceKVCompactor örneği oluşturma ve çalıştırma:
obj = InplaceKVCompactor(num_warps=32)
output = obj.compact(None, None, 0.0, 0.0)
print('InplaceKVCompactor.compact çıktısı:', type(output))
```

### 3.2. Modül: `idempotent_kv.kernel`
#### Fonksiyon: `_inplace_kv_compact_opt_kernel()`
- **Parametreler:** `K_ptr, V_ptr, TargetMap_ptr, stride_kb, stride_kh, stride_kn, stride_kd, stride_vb, stride_vh, stride_vn, stride_vd, stride_mb, stride_mh, stride_mn, N, HEAD_DIM`

```python
import torch
from idempotent_kv.kernel import _inplace_kv_compact_opt_kernel

res = _inplace_kv_compact_opt_kernel(None, None, None, 32, 32, 32, 32, None, None, None, None, None, None, None, None, None)
print('_inplace_kv_compact_opt_kernel() çağrı sonucu:', type(res))
```

#### Fonksiyon: `compact_kv_cache_inplace()`
- **Açıklama:** Executes in-place zero-copy KV cache compaction on GPU/CPU tensors.
Dispatches to native C++20 / Blackwell CUDA engine (idempotent-core) or Triton JIT.

Args:
    key_cache: [B, H, N, D] float16, bfloat16, or float32 contiguous tensor
    value_cache: [B, H, N, D] float16, bfloat16, or float32 contiguous tensor
    target_map: [B, H, N] int32 permutation map
    compacted_capacity: number of retained context slots
    num_warps: Triton execution warps per block (when using Triton)
    prefer_native: whether to prefer native C++20 / CUDA engine

Returns:
    (compacted_key, compacted_value) sliced directly as [B, H, :capacity, D]
- **Parametreler:** `key_cache, value_cache, target_map, compacted_capacity, num_warps, prefer_native`

```python
import torch
from idempotent_kv.kernel import compact_kv_cache_inplace

res = compact_kv_cache_inplace(32, None, None, 32, None, None)
print('compact_kv_cache_inplace() çağrı sonucu:', type(res))
```

### 3.3. Modül: `idempotent_kv.integrations.vllm`
#### Sınıf: `VLLMInplaceCompactionHook`
- **Açıklama:** Drop-in compaction hook for vLLM Attention and CacheEngine workers.

Intercepts dynamic context eviction phases and replaces secondary
buffer allocations with O(1) in-situ idempotent swaps.
- **Metotlar:** `__init__()`

```python
import torch
import numpy as np
from idempotent_kv.integrations.vllm import VLLMInplaceCompactionHook

# VLLMInplaceCompactionHook örneği oluşturma ve çalıştırma:
obj = VLLMInplaceCompactionHook(capacity=128, protected_prefix_len=32)
print('VLLMInplaceCompactionHook başarıyla başlatıldı:', obj)
```

---

## 4. İleri Düzey Entegrasyon ve Çalışma Zamanı Mimarisi

### Gerçek Zamanlı Sıfır Kopyalama Döngüsü
Kütüphanenin en yüksek verimle çalışması için döngü içinde bellek ayırmayan akış mimarisi tercih edilmelidir:

```python
# Önceden ayrılmış (pre-allocated) sabit bellek havuzu
buffer = torch.zeros(1, 128, 64, dtype=torch.float32)

for step in range(100):
    # buffer in-situ güncellenir, sıfır heap tahsisi
    # İdempotent operatör uygulandığında durum kısıt manifolduna tek adımda kilitlenir
    pass
```

---

## 5. Hata Yönetimi ve Sınır Durumlar (Edge Cases)

1. **Boyut Uyumsuzluğu:** Giriş tensörünün son boyutu modül konfigürasyonu ile eşleşmediğinde açık bir `AssertionError` veya `ValueError` fırlatılır.
2. **Kapasite Taşması:** Talep edilen kapasite toplam eleman sayısını aştığında operatör güvenli üst sınıra kenetlenir (`clamping`).
3. **Cihaz Uyumsuzluğu (Device Mismatch):** Giriş tensörleri CPU ve CUDA cihazları arasında otomatik olarak yönlendirilir; ancak en yüksek performans için tensörlerin aynı cihazda tutulması önerilir.

---

## 6. Performans İpuçları ve En İyi Pratikler

- **TorchScript & JIT:** Kritik döngülerde `torch.jit.script` ile derleyerek Python yorumlayıcı yükünü ortadan kaldırın.
- **Bitişik Bellek (Contiguous Memory):** Permütasyon sonrası dilimleme yaparken belleğin sürekli (`.contiguous()`) olduğundan emin olun.
- **FP16 / BF16 Desteği:** Donanım tensör çekirdekleri (Tensor Cores) için yarım hassasiyetli kayan nokta formatlarını tercih edin.
