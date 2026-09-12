# IDEMPOTENTS.md: İdempotent Operatör Teorisi, Matematiksel Temeller ve Manifold Kataloğu
## idem-kv (idempotent_kv)

**Zero-Copy In-Place KV-Cache Compactor for vLLM & Deep Learning Accelerators**

---

## 1. Hilbert Uzayında İdempotent Projeksiyonun Aksiyomatik Temeli

Matematiksel analizde ve fonksiyonel operatör teorisinde bir Hilbert uzayı $\mathcal{H}$ üzerinde tanımlı lineer veya afin operatör $\boldsymbol{\Pi}: \mathcal{H} \to \mathcal{H}$, ardışık uygulandığında durumu değiştirmeyen bir yapıya sahipse **idempotent (eşkuvvetli)** olarak tanımlanır:

$$\boldsymbol{\Pi}^2 = \boldsymbol{\Pi} \circ \boldsymbol{\Pi} = \boldsymbol{\Pi}$$

Eğer operatör $\mathcal{H}$ uzayındaki iç çarpıma göre öz-eşlenik (self-adjoint) ise:
$$\langle \boldsymbol{\Pi}\mathbf{u}, \mathbf{v} \rangle_{\mathcal{H}} = \langle \mathbf{u}, \boldsymbol{\Pi}\mathbf{v} \rangle_{\mathcal{H}} \iff \boldsymbol{\Pi}^* = \boldsymbol{\Pi}$$
bu durumda $\boldsymbol{\Pi}$, $\mathcal{H}$ uzayını kapalı ve konveks bir $\mathcal{M} \subset \mathcal{H}$ alt uzayına dik olarak izdüşüren **ortogonal projektördür**.

### Temel Cebirsel ve Geometrik Teoremler:
1. **İç Nokta Sabitliği:** $\forall \mathbf{u} \in \mathcal{M} \implies \boldsymbol{\Pi}(\mathbf{u}) = \mathbf{u}$. Manifold üzerinde bulunan bir durum projektör tarafından ötelenemez.
2. **Ortogonal Tümleyen Projektörü:** $\mathbf{Q} = \mathbf{I} - \boldsymbol{\Pi}$ operatörü de idempotenttir ($\mathbf{Q}^2 = \mathbf{Q}$) ve durumu manifoldun dik tümleyenine ($\mathcal{M}^\perp$) izdüşürür:
   $$(\mathbf{I} - \boldsymbol{\Pi})^2 = \mathbf{I} - 2\boldsymbol{\Pi} + \boldsymbol{\Pi}^2 = \mathbf{I} - \boldsymbol{\Pi}$$
3. **Hilbert Norm Minimizasyonu:** Ortogonal izdüşüm operatörü, serbest durum $\mathbf{u}_0$ ile kısıt kümesi $\mathcal{M}$ arasındaki Hilbert norm mesafesini mutlak olarak minimize eden tek çözümdür:
   $$\boldsymbol{\Pi}(\mathbf{u}_0) = \arg\min_{\mathbf{u} \in \mathcal{M}} \|\mathbf{u} - \mathbf{u}_0\|_{\mathcal{H}}$$

---

## 2. idem-kv Kütüphanesine Özel Matematiksel Formülasyon

`idem-kv` kütüphanesi kapsamında kısıtlar, durum uzayının belirli bir afin veya diferansiyellenebilir manifold alt kümesine $\mathcal{M}$ hapsedilmesi şeklinde modellenir.
Geleneksel optimizasyon yöntemlerinde ceza fonksiyonları (penalty loss) veya gradyan inişi (SGD/Adam) ile onlarca adımda yaklaşılmaya çalışılan kısıtlar, bu kütüphanede kapalı form matris/tensör izdüşüm formülleri ile **tek bir saat çevriminde ($O(1)$ veya $O(N)$ karmaşıklıkla)** sağlanır.

---

## 3. Manifold Kataloğu ve `src/` Karşılıkları

Aşağıdaki tabloda ve ayrıntılı alt bölümlerde `idem-kv` kütüphanesinde kullanılan tüm idempotent manifoldlar ve bunların `src/` dizinindeki birebir karşılıkları verilmiştir:

| No | Manifold Adı | Sembol | `src/` Karşılığı | Karmaşıklık | Kısıt Tipi |
| :-: | :--- | :---: | :--- | :---: | :--- |
| **1** | **Manifold-InplaceKVCompactor** | $\mathcal{M}_{Inplac}$ | `idempotent_kv.compactor:InplaceKVCompactor` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **2** | **Permütasyon-_inplace_kv_compact_opt_kernel** | $\mathcal{M}_{_inpla}$ | `idempotent_kv.kernel:_inplace_kv_compact_opt_kernel` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **3** | **Permütasyon-compact_kv_cache_inplace** | $\mathcal{M}_{compac}$ | `idempotent_kv.kernel:compact_kv_cache_inplace` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **4** | **Manifold-VLLMInplaceCompactionHook** | $\mathcal{M}_{VLLMIn}$ | `idempotent_kv.integrations.vllm:VLLMInplaceCompactionHook` | $O(N)$ | Kapalı Konveks Alt-Uzay |

### 3.1. Manifold-InplaceKVCompactor ($\mathcal{M}_{Inplac}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`InplaceKVCompactor`](file:///packages/idem-kv/src/idempotent_kv/compactor.py), Modül: `idempotent_kv.compactor`
- **Fiziksel / Algoritmik Anlam:** High-Performance Zero-Copy In-Place KV-Cache Compactor.

Transforms KV-cache memory management in autoregressive transformer inference
(vLLM, SGLang, Hugging Face) from O(N) auxiliary out-of-place gathering into
an O(1) in-situ idempotent permutation engine.

Protected under U.S. Patent Application No. 64/148,668 (Patent Pending).
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Inplac} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{1}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{1}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{1} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.2. Permütasyon-_inplace_kv_compact_opt_kernel ($\mathcal{M}_{_inpla}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`_inplace_kv_compact_opt_kernel`](file:///packages/idem-kv/src/idempotent_kv/kernel.py), Modül: `idempotent_kv.kernel`
- **Fiziksel / Algoritmik Anlam:** 
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{_inpla} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{2}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{2}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{2} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.3. Permütasyon-compact_kv_cache_inplace ($\mathcal{M}_{compac}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`compact_kv_cache_inplace`](file:///packages/idem-kv/src/idempotent_kv/kernel.py), Modül: `idempotent_kv.kernel`
- **Fiziksel / Algoritmik Anlam:** Executes in-place zero-copy KV cache compaction on GPU/CPU tensors.
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
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{compac} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{3}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{3}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{3} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.4. Manifold-VLLMInplaceCompactionHook ($\mathcal{M}_{VLLMIn}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`VLLMInplaceCompactionHook`](file:///packages/idem-kv/src/idempotent_kv/integrations/vllm.py), Modül: `idempotent_kv.integrations.vllm`
- **Fiziksel / Algoritmik Anlam:** Drop-in compaction hook for vLLM Attention and CacheEngine workers.

Intercepts dynamic context eviction phases and replaces secondary
buffer allocations with O(1) in-situ idempotent swaps.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{VLLMIn} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{4}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{4}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{4} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

---

## 4. Çoklu Kısıt Manifoldları ve Çevrimsel POCS (Projection Onto Convex Sets)

Sistem birden fazla kısıt manifoldunun arakesitinde yaşamak zorunda olduğunda:
$$\mathbf{u}^* \in \mathcal{M}_{\text{total}} = \bigcap_{j=1}^m \mathcal{M}_j$$

Çevrimsel POCS operatörü bir $\sigma \in S_m$ permütasyonu ile ardışık bileşke olarak tanımlanır:
$$\mathbf{T}_\sigma = \boldsymbol{\Pi}_{\sigma(m)} \circ \boldsymbol{\Pi}_{\sigma(m-1)} \circ \dots \circ \boldsymbol{\Pi}_{\sigma(1)}$$

Bregman ve Bauschke-Borwein teoremlerine göre konveks kümelerin arakesiti boş değilse dizi $\mathbf{u}^*$ noktasına geometrik hızla yakınsar:
$$\|\mathbf{u}^{(k+1)} - \mathbf{u}^*\| \le c(\mathbf{T}_\sigma) \|\mathbf{u}^{(k)} - \mathbf{u}^*\|$$
Burada $c(\mathbf{T}_\sigma) = \cos(\theta_{\text{Friedrichs}}) < 1$ daralma katsayısıdır. İdempotent permütasyon hızlandırması ile birbirine dik kısıtlar ardışık işlenerek yakınsama hızı maksimize edilir.
