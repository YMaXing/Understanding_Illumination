# From baseline_v0 to the default pipeline "tri ($\hat s$ after filtering) + veto, band factor 4"

Status as of 2026-10-27:

- **Default pipeline:** `configs/default.yaml`, identical to `configs/candidate_tri_veto4.yaml`. Stage B run name `tri_veto4`.
- **Original baseline:** `configs/baseline_v0.yaml`. It reproduces `stage_b/baseline_v0` bit-for-bit, and a test enforces this.

The task history is in `TASK_070_STRIPES.md` (T1–T5) and in the per-step reports under `baseline_output/diag/`.

## 1. What the new pipeline does differently

The baseline weight is

$$
w = \max\!\left(w_{\min},\ \frac{\varepsilon}{q_{\mathrm{pool}} + \varepsilon}\right),
\qquad
q_{\mathrm{pool}} = \operatorname{pool}_r\!\big(\max(q - \tau,\ 0)\big),
\qquad
q = \frac{\lvert L_\sigma(\rho)\rvert}{\max(\rho_\sigma,\ \rho_{\mathrm{floor}})},
$$

computed on DA3's disparity $\rho = 1/D$. $\operatorname{pool}_r$ is a max over a disk of radius $r$. Here $L_\sigma$ is the zero-sum Laplacian of Gaussian at $\sigma = 1.25$ px and $\rho_\sigma$ is the Gaussian-smoothed disparity. The new pipeline changes **how $q$ is computed** and nothing else:

| step | baseline_v0 | new default |
|---|---|---|
| Disparity fed to the Laplacian | $\rho = 1/D$ as output by DA3 | **$\rho_f$ = tangent-plane ("trilateral") bilateral filter of $\rho$** |
| Response | $q = \lvert L_\sigma(\rho)\rvert / \max(\rho_\sigma, \rho_{\mathrm{floor}})$ | **$q' = \min\!\big(q_f,\ g\,\lvert L_\sigma(\rho_f) - L_{4\sigma}(\rho_f)\rvert / \max(\rho_{\sigma,f}, \rho_{\mathrm{floor}})\big)$**, with $q_f$ from $\rho_f$ and $g = 1/(1 - 1/4) = 4/3$ |
| Noise scale $\hat s$ | $\operatorname{median}(q)/0.6745$ on clean pixels | the same estimator, applied to the **filtered** (non-veto) response $q_f$ |
| Threshold, falloff, pooling, $w$, upsampling, frames | | unchanged: $\tau = 3\hat s$, $\varepsilon = 3\hat s$, max-pool over a disk of radius $r = 3$, $w_{\min} = 0.05$ |

The three changes:

1. **Trilateral prefilter** (`detector.prefilter: trilateral`), which removes DA3's grid ripple.
   - At each pixel it averages neighbours' **deviation from the local tangent plane**, $e(x,d) = \rho(x+d) - \rho(x) - \nabla\rho(x)\cdot d$.
   - Weights: a spatial Gaussian ($\sigma_s = 3$ px) × a range Gaussian on $e/\rho(x)$ ($\sigma_r = 1\%$). Sky neighbours are excluded. The gradient comes from a Gaussian-derivative kernel with $\sigma_g = 3$ px.
   - Properties:
     - Planes are reproduced exactly, except within about 9 px of an edge or the sky (section 2.4).
     - Invariant to the global depth scale.
     - Occlusion edges and sharp folds (large $e$) get no weight, so they are kept.
     - DA3's 7-px ripple ($e/\rho \approx 0.1$–$0.5\%$) is averaged out.
   - Code: `illum_baseline/detector.py: trilateral, gaussian_gradient`. Full explanation and math: section 2.
2. **$\hat s$ after filtering** (`detector.tri_shat: filtered`). The ripple had been inflating $\hat s$, so measuring noise after the filter lowers $\hat s$ by roughly $2.5\times$, and with it the threshold $\tau = 3\hat s$. Weak folds, such as the lit/shaded edges of boxes, then get a stronger response.
3. **Scale-ratio veto** (`detector.scale_band: true`, `scale_band_mode: veto`, `band_factor: 4`), which keeps smooth curved surfaces from being flagged.
   - Smooth constant curvature (a large cylinder or sphere) gives the same Laplacian at every scale, so $L_\sigma - L_{4\sigma}$ cancels it exactly.
   - Folds, steps and fine texture all respond more strongly at the smaller scale, so they pass the veto. The gain $g = 1/(1-1/4)$ restores an ideal fold's band response, on the fold line, to its full value.
   - Without the veto, the lower threshold from change 2 flags curved objects. In 083 the bollard's median $w$ was 0.58 without the veto and is 1.00 with it.

Everything else in the baseline is unchanged: preprocessing, DA3 and GeoCalib inference, the sky mask, the kernels, the threshold-then-pool order, and the output format and frames. The new pipeline needs **no extra DA3 passes**; the filter is post-processing of the existing depth.

## 2. The trilateral prefilter in detail

### 2.1 Background: Gaussian, bilateral and trilateral filters

All three filters replace a pixel's value with a weighted average of its neighbours. They differ in how the weights are chosen. Below, $\rho$ is the disparity, $x$ a pixel, and $d$ an integer offset to a neighbour.

- **Gaussian filter.** The weights depend only on distance:

  $$
  \rho_G(x) = \frac{\sum_d G_s(d)\,\rho(x+d)}{\sum_d G_s(d)},
  \qquad
  G_s(d) = \exp\!\left(-\frac{\lVert d\rVert^2}{2\sigma_s^2}\right).
  $$

  It is linear and shift-invariant. It removes small wiggles, but it averages across edges, so depth steps and folds get blurred.
- **Bilateral filter** (Tomasi & Manduchi, ICCV 1998). It adds a *range* weight $G_r$ on how different the neighbour's value is:

  $$
  \rho_B(x) = \frac{\sum_d G_s(d)\,G_r\big(\rho(x+d) - \rho(x)\big)\,\rho(x+d)}{\sum_d G_s(d)\,G_r\big(\rho(x+d) - \rho(x)\big)}.
  $$

  Neighbours across a large jump get almost no weight, so edges stay sharp. Implicitly, though, it treats the signal as **piecewise constant**. A neighbour on a *slope* differs from the centre by $g\cdot d$, where $g$ is the slope, even with no edge between them. This has two consequences:
  - on a steep slope, the range weight cuts the neighbourhood short along the slope direction;
  - wherever the weights are one-sided (next to an edge), the average of a slope is biased by $g\cdot\mathbb{E}_w[d] \neq 0$. That would put a false response on slanted planes.
- **Trilateral filter** (Choudhury & Tumblin, *The trilateral filter for high contrast images and meshes*, EGSR 2003). It "tilts" the range kernel along the local gradient, so it treats the signal as **piecewise linear** instead. The name refers to three weighting ingredients: spatial distance, range, and the local tilt (gradient).

**Our filter is a simplified tangent-plane version of that idea.** Choudhury & Tumblin also bilaterally filter the gradient field itself and adapt the neighbourhood size. We estimate the gradient once with a Gaussian-derivative kernel and measure the range term as the deviation from the tangent plane that gradient defines. This keeps the filter simple and exactly linear-preserving, which matters because the detector must give exactly zero on planes.

### 2.2 Why this problem needs it

- **The detector responds strongly to DA3's ripple.** The stripes are a ripple in DA3's disparity locked to its pixel grid, with a period of $\lambda \approx 7$ px and a relative amplitude $a \approx 0.1$–$0.5\%$ (T1). The detector's Laplacian ($\sigma = 1.25$ px) responds to a relative sinusoid $\rho \propto 1 + a\sin(kv)$, with $k = 2\pi/\lambda$, with

  $$
  q \approx a\,k^2\,e^{-\sigma^2 k^2/2}.
  $$

  At $\lambda = 7$ px the gain $k^2 e^{-\sigma^2k^2/2}$ is $0.43\ \mathrm{px}^{-2}$, 91% of its maximum. The maximum is at $k = \sqrt2/\sigma$, i.e. $\lambda = \pi\sqrt2\,\sigma \approx 5.6$ px. So $a = 0.2\%$ already gives $q \approx 8.6\times10^{-4}$, about $11\,\hat s$ on 070 ($\hat s = 7.8\times10^{-5}$). That is far above the threshold $\tau = 3\hat s$, which explains the stripes in $w$.
- **A plain Gaussian (or a larger $\sigma$) would remove the ripple but damage everything else.** It blurs steps and folds, and the task explicitly warns that a larger $\sigma$ makes longer-wavelength coherent ripples *stronger* relative to $\hat s$.
- **A plain bilateral filter fails exactly where the stripes are worst.** On the grazing floor of 070, disparity changes by about $\lvert\partial_v\rho\rvert/\rho \approx 1\%$ *per pixel* (at $v \approx 300$). Neighbours 3 px away differ by about $3\%$ from the slope alone.
  - A range kernel tight enough to protect real edges therefore cuts off after 2–3 px along $v$, so it cannot average a 7-px ripple.
  - A range kernel loose enough to average it degenerates into a Gaussian.
  - Its one-sided bias near edges would also break the detector's exact zero on planes (test 5).
- **The alternatives we tried were worse** (section 4):
  - latin14 TTA removed 86% of the ripple, but costs 28 DA3 passes per image and halves $\hat s$;
  - the $\tau$ floor hides the stripes without removing them, and weakens folds;
  - the trilateral filter is post-processing of the existing depth.

### 2.3 Definition (as implemented in `detector.py: trilateral`)

Let $S$ be the sky mask, $\Omega = \lnot S$ the non-sky pixels, and $\rho = 1/D$ on $\Omega$ ($\rho = 0$ on $S$). The parameters are $\sigma_s = 3$ px, $\sigma_r = 0.01$ and $\sigma_g = 3$ px.

1. **Tangent-plane gradient.** Separable Gaussian-derivative kernels on $t \in \lbrace -R_g, \dots, R_g \rbrace$, with $R_g = \lceil 4\sigma_g\rceil = 12$ and odd-reflect padding:

   $$
   g_0(t) = \frac{e^{-t^2/2\sigma_g^2}}{\sum_{t'} e^{-t'^2/2\sigma_g^2}},
   \qquad
   k_1(t) = \frac{t\,g_0(t)}{\sum_{t'} t'^2\,g_0(t')},
   $$

   $$
   g(x) = \begin{pmatrix} g_u(x) \\ g_v(x) \end{pmatrix},
   \qquad
   g_u(u, v) = \sum_{t, t'} k_1(t)\,g_0(t')\,\rho(u+t,\ v+t'),
   \qquad
   g_v(u, v) = \sum_{t, t'} g_0(t)\,k_1(t')\,\rho(u+t,\ v+t'),
   $$

   in disparity per pixel along $(u, v)$. These are *correlations* (neighbour at $+t$ weighted by $k_1(t)$). Because $k_1$ is antisymmetric, the same sum written as a convolution would return $-\nabla\rho$. Because $\sum_t k_1(t) = 0$ and $\sum_t t\,k_1(t) = 1$ (and $\sum_t g_0(t) = 1$), the estimate is exact on affine functions: on $\rho = a + b_u u + b_v v$ it returns $(b_u, b_v)$ wherever its support lies on the plane, borders included, since odd-reflect padding preserves affine functions.
2. **Deviation from the tangent plane** for each offset $d$ in the disk $\lVert d\rVert \le R_s = \lceil 3\sigma_s\rceil = 9$ (253 offsets; neighbours outside the image use odd-reflect padding):

   $$
   e(x, d) = \rho(x+d) - \rho(x) - g(x)\cdot d.
   $$

   This is how far the neighbour lies off the plane that touches $\rho$ at $x$.
3. **Weights:** spatial × range × not-sky:

   $$
   w(x, d) = \exp\!\left(-\frac{\lVert d\rVert^2}{2\sigma_s^2}\right)\,
   \exp\!\left(-\frac{1}{2}\left(\frac{e(x,d)}{\sigma_r\,\max\big(\rho(x),\,\rho_{\mathrm{floor}}\big)}\right)^{2}\right)\,
   \mathbf{1}\big[x+d \notin S\big].
   $$

   The range term is *relative* ($e$ divided by the local disparity), which makes the filter independent of depth scale. $\rho_{\mathrm{floor}}$ is the same floor the detector uses ($0.5\times$ the 1st percentile of $\rho$ on $\Omega$), so the term stays finite near the sky.
4. **Output.** For $x \in \Omega$ (sky pixels keep $\rho_f = 0$):

   $$
   \rho_f(x) = \rho(x) + \frac{\sum_d w(x,d)\,e(x,d)}{\sum_d w(x,d)}
   = \frac{\sum_d w(x,d)\,\big[\rho(x+d) - g(x)\cdot d\big]}{\sum_d w(x,d)}.
   $$

   - The second form shows the idea: each neighbour's value is first *transported back to $x$ along the tangent plane*, $\rho(x+d) - g(x)\cdot d$, and those transported values are averaged.
   - The centre term ($d = 0$) has $e = 0$ and $w = 1$, so the denominator is $\ge 1$.

### 2.4 Properties (with short derivations)

| property | why | consequence |
|---|---|---|
| **Planes are reproduced exactly (away from edges)** | Let $\rho(y) = c_0 + b\cdot y$ on every neighbour of $x$ within $R_s$. Then $e(x,d) = (b - g(x))\cdot d$, which is *odd* in $d$, while the weights depend on $e^2$ and $\lVert d\rVert^2$ and are *even*. So $\sum_d w\,e = 0$ and $\rho_f(x) = \rho(x)$, **even if the gradient estimate $g(x)$ is wrong**. If the gradient's support also lies on the plane, $g = b$, $e \equiv 0$, and exactness holds for any weights. | The detector's exact zero on planes of any slant (test 5) is kept; a plain bilateral filter would bias slopes. Tested: `test_trilateral_planes_exact`. Synthetic floor-and-box scene: $\lvert\rho_f-\rho\rvert/\rho \le 2.3\times10^{-16}$ at every plane pixel more than 9 px from a feature. |
| **Independent of depth scale** | $\rho \to c\rho$ scales $g$, $e$, $\rho(x)$ and $\rho_{\mathrm{floor}}$ by $c$, so every weight is unchanged and $\rho_f \to c\rho_f$. $q$ is normalized by $\rho$, so it is unchanged. | Test 8 is kept (`test_trilateral_scale_invariance`). |
| **The ripple is averaged out** | For a plane with a relative ripple $a\sin(kv)$ (and a ripple-free tangent plane), $\lvert e\rvert/\rho \lesssim 2a \approx 0.2$–$1\%$, which is at or below $\sigma_r$. The range weights are then close to 1 ($\ge e^{-1/2} \approx 0.61$), and the filter acts approximately like a Gaussian of $\sigma_s = 3$ px. That attenuates a wavenumber $k$ by $e^{-\sigma_s^2k^2/2}$: **0.027 ($\approx 38\times$) at $\lambda = 7$ px**. The tangent plane is ripple-free too, because the $\sigma_g = 3$ px derivative kernel suppresses the ripple just as strongly. | Measured on 070: the 7.1-px stripe amplitude falls from $1.92\times10^{-4}$ to $5.6\times10^{-6}$ (−97%). Tested: `test_trilateral_attenuates_7px_ripple_and_keeps_features`. |
| **Steps are kept sharp** | Across an occlusion, $e \approx \Delta\rho$, typically 20–100% of $\rho$, so the range weight is $\approx \exp\!\Big(-\tfrac12\big(\tfrac{\Delta\rho}{\sigma_r\,\rho}\big)^2\Big) \approx 0$ and the far side contributes nothing. | Silhouettes stay sharp; edge retention is 1.000 on all images. In the synthetic scene the peak response at steps is unchanged ($q \approx 0.21$). |
| **Folds are partly kept** | Across a fold with slope change $\Delta s$, a neighbour at distance $t$ past the fold has $e \approx \Delta s\,t$. For the box folds ($\Delta s/\rho$ between about $1/f_n$ and $2/f_n$, i.e. $0.3$–$0.5\%$ per px at $f_n \approx 366$ px; an estimate), $e/\rho$ reaches about $0.6$–$1.5\%$, i.e. $\approx \sigma_r$, at 2–3 px. Those neighbours are down-weighted but not removed, so the fold is slightly rounded. | Fold responses drop somewhat: on 079, $q/\tau$ goes from 2.0–2.7 to 1.4–1.6 at the baseline threshold. This is why $\hat s$ is estimated after filtering (change 2). |
| **Fine texture is smoothed** | Grass and gravel deviate from their local planes by about the ripple's size. | 079 grass with $w < 0.5$: 0.60 → 0.22 (with $\hat s$ from the unfiltered response). |
| **Small bias next to edges** | Within $R_s = 9$ px of a step, fold or the sky, the window is one-sided: neighbours across the feature get no weight. The gradient estimate is also contaminated there (support $R_g = 12$ px), so the same-side deviations $(b - g)\cdot d$ no longer cancel. $\rho_f$ is then slightly biased. | Synthetic floor-and-box scene: $\lvert\rho_f-\rho\rvert/\rho$ up to $4.7\times10^{-3}$ within 6 px of a feature, $1.3\times10^{-4}$ at 6–9 px, exactly 0 beyond 9 px. The resulting spurious response on the adjacent plane reaches $q \approx 1.7\times10^{-4}$ at 6–9 px (baseline: 0), $1.5\times10^{-5}$ at 9–13 px, and is negligible beyond. That is comparable to $\tau$ ($\approx 0.7$–$2.4\times10^{-4}$), so down-weighted bands next to strong edges can widen by a few pixels. |

### 2.5 How it changes the response mathematically

**Baseline_v0.** With the custom zero-sum kernels $g_0$ (Gaussian) and $g_2$ (second derivative) at $\sigma = 1.25$ px (`BASELINE_SPEC.md`, M3 step B3):

$$
\rho_\sigma = (g_0 \otimes g_0) \ast \rho,
\qquad
L = L_\sigma(\rho) = (g_2 \otimes g_0 + g_0 \otimes g_2) \ast \rho,
\qquad
q = \frac{\lvert L\rvert}{\max(\rho_\sigma,\ \rho_{\mathrm{floor}})}.
$$

**New pipeline.** The same operators are applied to the filtered disparity. $\rho_{\mathrm{floor}}$ is still computed from the unfiltered $\rho$.

$$
\rho_{\sigma,f} = (g_0 \otimes g_0) \ast \rho_f,
\qquad
L_f = L_\sigma(\rho_f),
\qquad
q_f = \frac{\lvert L_f\rvert}{\max(\rho_{\sigma,f},\ \rho_{\mathrm{floor}})},
$$

$$
q' = \min\!\left(q_f,\ \ g\,\frac{\lvert L_\sigma(\rho_f) - L_{4\sigma}(\rho_f)\rvert}{\max(\rho_{\sigma,f},\ \rho_{\mathrm{floor}})}\right),
\qquad g = \frac{1}{1 - 1/4} = \frac{4}{3}
\qquad \text{(scale-ratio veto, section 1; } L_{4\sigma} \text{ uses the same zero-sum kernels at } 4\sigma = 5\text{ px)},
$$

$$
\hat s = \frac{\operatorname{median}_{\text{clean pixels}}(q_f)}{0.6745}
\qquad \text{("}\hat s\text{ after filtering")},
$$

$$
\tau = 3\hat s,\quad \varepsilon = 3\hat s,\qquad
w = \max\!\left(w_{\min},\ \frac{\varepsilon}{\operatorname{pool}_r\!\big(\max(q' - \tau,\ 0)\big) + \varepsilon}\right)
\qquad \text{(unchanged)}.
$$

**What this amounts to.** Write $\rho_f = \rho + \delta$, with

$$
\delta(x) = \frac{\sum_d w(x,d)\,e(x,d)}{\sum_d w(x,d)}.
$$

Because the Laplacian is linear, $L_f = L + L_\sigma(\delta)$. Two regimes:

- **Smooth regions** (all deviations $\ll \sigma_r$): the weights reduce to the spatial Gaussian, and the tangent term cancels by symmetry ($\sum_d G_s(d)\,d = 0$). So $\rho_f \approx G_s \ast \rho$, and because Gaussians compose:

  $$
  L_f \approx L_\sigma(G_s \ast \rho) = L_{\sigma_{\mathrm{eff}}}(\rho),
  \qquad
  \sigma_{\mathrm{eff}} = \sqrt{\sigma^2 + \sigma_s^2} = \sqrt{1.25^2 + 3^2} \approx 3.25\ \mathrm{px}.
  $$

  The detector behaves like a Laplacian at **3.25 px** there. Its response to a relative sinusoid becomes $a\,k^2 e^{-\sigma_{\mathrm{eff}}^2 k^2/2}$ instead of $a\,k^2 e^{-\sigma^2 k^2/2}$:

  | wavelength $\lambda$ | baseline gain ($\sigma = 1.25$) | new gain in smooth regions ($\sigma_{\mathrm{eff}} = 3.25$) | reduction |
  |---|---|---|---|
  | 7 px (DA3 grid ripple) | 0.429 | 0.0114 | **$38\times$** |
  | 15 px | 0.153 | 0.070 | $2.2\times$ |
  | 21 px (slow floor undulation) | 0.081 | 0.055 | $1.5\times$ |

  Short-wavelength artifacts are strongly suppressed. Long-wavelength content, which responds weakly anyway, is nearly unchanged.
- **At discontinuities** (deviations $\gg \sigma_r$): the far side gets no weight, so $\rho_f$ stays close to $\rho$. The bias is at most about 0.5% of $\rho$ within 6 px (property table). The detector keeps its original **1.25-px** scale and sharp localization, and step responses are essentially unchanged. The only side effect is a weak spurious response up to about 9 px from strong edges (property table).

So the trilateral prefilter turns the fixed-scale Laplacian into an **edge-adaptive** one: coarse ($\sigma_{\mathrm{eff}} \approx 3.25$ px) inside smooth surfaces, where the ripple and texture live, and fine ($\sigma = 1.25$ px) at real discontinuities.

- **Folds sit in between:** they are partly smoothed (property table).
- **Consequences for $\hat s$ and the threshold:** most pixels are in smooth regions, so the median of $q_f$, and hence $\hat s$, falls to 0.28–0.69 times its unfiltered value across the 10 images. Using that $\hat s$ (change 2) lowers $\tau$, which restores the folds' margin. A lower $\tau$ would also catch smooth curvature: for a quadratic surface, $L_\sigma(\rho)$ is the same at every $\sigma$, so the $\sigma_{\mathrm{eff}}$-scale Laplacian sees it just as the baseline does. The scale-ratio veto (change 3) removes that, because $L_\sigma - L_{4\sigma} = 0$ on any quadratic.

### 2.6 Parameters and sensitivity

| parameter | value | role | effect of changing it |
|---|---|---|---|
| `tri_sigma_s` | 3 px | spatial scale $\sigma_s$; sets $\sigma_{\mathrm{eff}}$ and the ripple attenuation $e^{-\sigma_s^2k^2/2}$ | must be $\gtrsim 2.5$ px to suppress a 7-px ripple; larger values blur folds more and extend the support ($R_s = \lceil 3\sigma_s\rceil$) |
| `tri_sigma_r` | 1% | $\sigma_r$: the relative deviation from the tangent plane that still gets full weight; separates ripple and texture ($\lesssim 0.5\%$) from steps and folds ($\gtrsim 1$–$1.5\%$ at 2–3 px) | tighter values keep folds slightly better but let texture back in (table below); the ripple stays removed down to 0.3% |
| `tri_sigma_g` | 3 px | $\sigma_g$: scale of the tangent-plane gradient | must suppress the ripple in the gradient ($\gtrsim 2.5$ px); larger values make the tangent plane wrong on curved surfaces |
| `tri_shat` | `filtered` | where $\hat s$ is estimated | `unfiltered` keeps the baseline threshold: less down-weighting, but folds weaker ($w$ 0.59–0.68 on 079) |

Measured sensitivity to $\sigma_r$, with $\hat s$ from the unfiltered response and no veto:

| $\sigma_r$ | 079 fold $w$ | 079 grass, share $w<0.5$ | 070 floor, share $w<0.9$ | 070 7.1-px amplitude |
|---|---|---|---|---|
| 1% (chosen) | 0.59–0.68 | 0.22 | 0.008 | $5.6\times10^{-6}$ |
| 0.5% | 0.48–0.57 | 0.34 | 0.003 | $6.9\times10^{-6}$ |
| 0.3% | 0.44–0.53 | 0.43 | 0.003 | $9.8\times10^{-6}$ |

A tighter $\sigma_r$ helps folds less than it lets grass back in, so the fold-to-texture separation is best at 1%.

**Cost:** 253 shifted-array operations per image, plus the two Laplacians at $\sigma$ and $4\sigma$. This runs in numpy on the CPU, with no extra DA3 passes. Measured: about 0.46 s per image for the whole detector (baseline_v0: 0.02 s), negligible next to DA3 and GeoCalib.

## 3. Results (10 images; baseline = `baseline_v0`)

| image | share $w<0.5$, baseline → new | $\Delta$ share $w<0.5$ | $\Delta$ share $w<0.9$ | edge retention |
|---|---|---|---|---|
| 070 | 0.239 → 0.215 | −0.024 | −0.061 | 1.000 |
| 075 | 0.307 → 0.286 | −0.021 | −0.042 | 1.000 |
| 076 | 0.249 → 0.224 | −0.025 | −0.039 | 1.000 |
| 079 | 0.303 → 0.216 | −0.087 | −0.119 | 1.000 |
| 080 | 0.230 → 0.215 | −0.015 | −0.053 | 1.000 |
| 082 | 0.257 → 0.253 | −0.004 | −0.031 | 1.000 |
| 083 | 0.153 → 0.159 | +0.006 | −0.054 | 1.000 |
| 084 | 0.335 → 0.271 | −0.065 | −0.098 | 1.000 |
| 088 | 0.112 → 0.105 | −0.008 | −0.068 | 1.000 |
| 090 | 0.367 → 0.332 | −0.036 | −0.061 | 1.000 |

- **Over all 10 images:** median $\Delta$ share of $w<0.5$ is −0.022, median $\Delta$ share of $w<0.9$ is −0.058, there are no regressions (none rises by more than 1 point), and edge retention is 1.000 everywhere. Edge retention means pixels with $q/\hat s > 100$ in the baseline that stay at $w < 0.5$.
- **070 floor stripes:**

| | baseline | new |
|---|---|---|
| floor, share $w<0.9$ | 0.275 | 0.031 |
| 7.1-px stripe amplitude in the floor residual | 1.92e-4 | 5.6e-6 (−97%) |

- **Spot checks:**

| check | baseline | new |
|---|---|---|
| 079 lit/shaded box fold, $w$ | 0.35–0.45 | 0.50–0.59 |
| 079 grass, share $w<0.5$ | 0.60 | 0.40 |
| 083 bollard surface, median $w$ | 1.00 | 1.00 |

The comparison reports are `baseline_output/diag/veto4/veto4_report.md`, with per-image weight stacks in `veto4/<stem>_weights.png`, and `baseline_output/diag/070/veto4_metrics.md`.

## 4. Important decisions and the evidence behind them

**Inherited from M0–M5 (still in force)**

- **DA3 runs with batch size 1.** Its bf16 numerics depend on batch composition: depth changed by up to 5% at isolated pixels between batch sizes.
- **Sky:** DA3's sky output is a ReLU score, not a probability. It is clipped to [0, 1] and thresholded at 0.5.
- **Preprocessing from the M1 audit:** black level 0, white level 65535, `white_balance: none` (the data are already white-balanced), exposure P99 → 0.9.
- **Focal mode `per_image`.** Images come from mixed cameras and zoom settings with no EXIF, so the pooled focal is recorded but flagged as not meaningful.
- **GeoCalib is reseeded before every call.** Its `NMF2D` draws random bases on each forward pass, which gave about ±1% spread in f.

**Diagnosing the floor stripes (T1)**

- **The stripes are a DA3 artifact locked to its pixel grid,** with a period of about 7 px, half the 14-px patch. The evidence:
  - their wavelength is constant down the image, whereas a pattern on the ground would scale with perspective (by 3.3× across the floor);
  - the stripe pattern moves with DA3's grid when the input is shifted by 4 px (the test is done with $L/\rho$; a 7-px shift cannot tell grid from scene for a 7-px ripple).
- **Posterization of the 8-bit input is not the cause.** The spec'd code-transition test was saturated, and the smoothed-luma version showed no association.

**Approaches tried and set aside (each remains available as an option, off by default)**

| approach | outcome | why not adopted |
|---|---|---|
| Input-side: `exposure_mode: median_rolloff`, `dither: tpdf` (T2) | floor codes 44 → 71, stripes unchanged | wrong cause: the artifact is grid-locked, not posterization |
| `da3_float_input` (T2) | not implemented | DA3 converts any array input to 8-bit RGB |
| TTA, spec shifts (0/7 px) (T3) | stripes unchanged | shifts that are multiples of 7 px leave a 7-px ripple in place |
| TTA `latin14` + 14-px feathering (T3) | stripe amplitude −86%, border frame fixed | halves $\hat s$, so texture down-weighting rises; costs 28 DA3 passes per image |
| $\tau$ floor `a_min` (T4) | hides the stripes (floor clean at 0.5%) | hides rather than removes; becomes a fixed threshold; weakens folds |
| $\varepsilon$ floor `eps_floor` (a user-approved departure from the task) | A0LF + 0.5% + ε floor: 0/10 regressions, −7.2 points | folds too weak, $w$ 0.69–0.81 on the lit/shaded box folds the user flagged |
| tri with $\hat s$ before filtering | 0/10 regressions, −7.9 points | folds weaker still (0.59–0.68) |
| tri with $\hat s$ after filtering, no veto | folds strong (0.30–0.34) | curved surfaces flagged (083 bollard median $w$ 0.58); 3/10 regressions |
| Scale band used directly as q (`scale_band_mode: replace`) | bollard fixed | texture passes while folds lose half their response: 10/10 regressions |
| Veto with band factor 2 | bollard fixed, 0 regressions | DA3 rounds folds over a few px, which a $\times 2$ ratio treats as curvature (fold $w$ 0.65–0.84) |
| Veto with band factor 6 | folds 0.44–0.51 | the larger kernel reaches silhouettes, so the bollard creeps back (17% below $w = 0.5$), 1 regression |

**Why \"tri ($\hat s$ after) + veto, factor 4\" was chosen:** it is the only variant that meets all of these at once:

- the stripes are removed at the depth level;
- there are no regressions on any image, and edge retention is full;
- curved surfaces are not flagged;
- folds are stronger than with the other no-regression variants;
- it adds no DA3 cost.

**Defaults changed in the YAML only.** `configs/default.yaml` now carries the new settings, but the code's built-in dataclass defaults remain the baseline's. Run snapshots made before these options existed, such as `baseline_v0`'s, are completed with the built-in defaults when they are reloaded. Changing the built-ins would silently turn old runs into the new pipeline.

**Every new behaviour is behind an option whose built-in default is the old behaviour.** `baseline_v0` reproduces bit-for-bit, including its JSON records, and `tests/test_baseline_repro.py` checks this.

## 5. Known limitations and open items

- **Folds (normal discontinuities) are still slightly weaker than in the baseline** (079 fold $w$ 0.50–0.59 vs 0.35–0.45). Within this detector, fold responses overlap with texture, and DA3 rounds folds over a few pixels. The designed remedy is a **surface-normal-angle channel**, with $w = \min(w_{\mathrm{Laplacian}},\ w_{\mathrm{angle}})$; it is on hold.
- **The trilateral filter also smooths genuine micro-relief** that deviates less than about 1% from the local plane.
- **Edge retention only checks strong edges** ($q > 100\,\hat s$). Weak geometry has been checked visually only. A small annotated fold set (e.g. made with `point-annotator`) would make this measurable.
- **The evaluation set is 10 images.** The percentile-based QA flags (P2/P98) are not meaningful at this size.
- **Folds and the bollard were measured at hand-picked image regions** (079 rows 230/254; 083 u 175–205, v 260–330).

## 6. Inference on new raw images (1023×768 or 768×1023)

### 6.1 Requirements

- Run from `codebase/geometry/` in the project's uv environment:
  - install with `env -u PYTHONPATH uv sync`;
  - always run Python as `env -u PYTHONPATH uv run ...`, because ROS Jazzy on `PYTHONPATH` otherwise leaks into imports.
- **GPU:** CUDA is used when available. Otherwise the pipeline falls back to CPU with a warning, which is much slower.
- **Model weights download on first use:** DA3METRIC-LARGE from the Hugging Face Hub (about 1.3 GB) and GeoCalib `pinhole` from GitHub releases (about 116 MB).
- **Input format** (the formats the preprocessing config was confirmed for, in the M1 audit):
  - **16-bit linear RGB TIFFs** (3 channels, already demosaiced; any TIFF compression), **1023×768 (landscape) or 768×1023 (portrait)**;
  - already black-subtracted (black level 0), white level 65535 ($2^{16}-1$), already white-balanced;
  - EXIF Orientation is respected if present.
- **If your data differ** (another camera pipeline, an 8-bit or gamma-encoded input, a raw mosaic, a black-level offset): run the audit first, `python -m illum_baseline audit -i /path/to/tiffs -o /path/to/outputs`, and adjust `preprocess:` in a copy of the config. Other resolutions work mechanically but are untested; the network size is always long side 504 px with both sides multiples of 14.
- **File stems must be unique** within the input folder, including its subfolders; the search is recursive (`**/*.tif*`).
- **The output directory must not be inside the input directory.**

### 6.2 Run (one command)

```bash
cd "/path/to/codebase/geometry"

# Paths from the command line (highest precedence):
env -u PYTHONPATH uv run python -m illum_baseline infer --input /path/to/my_raw_tiffs --output /path/to/my_outputs

# ... or from the YAML: edit paths.data_dir / paths.out_dir in configs/default.yaml (or a copy), then just:
env -u PYTHONPATH uv run python -m illum_baseline infer                 # uses configs/default.yaml
env -u PYTHONPATH uv run python -m illum_baseline infer --config configs/my_project.yaml
```

`infer` runs, for every TIFF under the input folder:

1. Stage A: preprocessing, the GeoCalib focal length, and DA3 depth and sky.
2. Stage B: the detector, the weights, the manifest and the diagnostic panels.

It then prints a JSON summary with the output locations. Both stages are cached, so rerunning the same command only processes new images.

**Where the input and output folders come from** (later entries override earlier ones):

| source | example | relative paths are relative to |
|---|---|---|
| YAML `paths.data_dir`, `paths.out_dir` (`--config`, default `configs/default.yaml`) | `data_dir: ../data` | the YAML file's own folder (so `../data` in `configs/default.yaml` is `geometry/data`) |
| `--set paths.data_dir=… --set paths.out_dir=…` | `--set paths.out_dir=/scratch/out` | the current working directory |
| `--input/-i DIR`, `--output/-o DIR` | `-i ./tiffs -o ./out` | the current working directory |

`~` is expanded everywhere. The resolved absolute paths are what the run records in `config_resolved.yaml`.

**Useful flags** (`python -m illum_baseline infer --help` lists them all):

| flag | effect |
|---|---|
| `--stems a b c` | only these images (file names without extension) |
| `--force` | recompute images whose outputs already exist |
| `--qa` | also write the dataset QA report (`qa_report.md`) |
| `--no-viz` | skip the diagnostic panels (faster) |
| `--run-name NAME` | Stage B output subfolder (default `run.name`, i.e. `tri_veto4`) |
| `--config FILE` | another pipeline config, e.g. `configs/baseline_v0.yaml` for the original baseline; may be given before or after `infer` |
| `--set KEY=VALUE` | any config value, e.g. `--set device=cpu`, `--set detector.k_tau=4` (repeatable) |
| `--k-tau 4`, `--sky-threshold 0.6`, … | detector and sky parameters (same as `--set detector.…`) |

Notes:

- **Exit code 2 with a short message** means the input folder is missing or empty, a requested stem doesn't exist, or the output folder is inside the input folder. Nothing is computed in that case.
- **DA3's per-call `[INFO]` logging is silenced** (warnings still show). Set `DA3_LOG_LEVEL=INFO` to see it.
- **Every subcommand** (`stage-a`, `stage-b`, `viz`, `qa`, `audit`, the `diag-*` tools) accepts the same `--config`, `--set` and `--input/--output` flags. For example, re-render panels only with `python -m illum_baseline viz -o /path/to/my_outputs`.
- **Mixing inputs in one output folder:** the pooled-focal QA value (`focal.f_pooled`, flagged as not meaningful, never used for the weights) is a median over *all* images cached in that output folder, and it is rewritten in each image's Stage A JSON whenever new images are added. Use one output folder per dataset if you want those records to stay fixed.
- **Checked:** `infer -i … -o … --qa` on a copy of one dataset TIFF in a fresh folder outside the project produced `w`, and every other array, bit-identical to the in-project run of the same image.

### 6.3 Outputs

```
$OUT/stage_a/<stem>.npz            D (canonical depth, float32) and sky_prob (float16); network res, UPRIGHT frame
$OUT/stage_a/<stem>.json           preprocessing, GeoCalib focal and gravity, DA3 metadata, provenance
$OUT/stage_a/srgb/<stem>.png       upright full-res sRGB used for GeoCalib and the viz
$OUT/stage_b/tri_veto4/<stem>.npz  STORED frame (aligned with the TIFF):
                                     w           float16, original resolution: the deliverable weight in [0.05, 1]
                                     w_net, q, L_over_rho   float16, network resolution
                                     D           float32, network resolution
                                     sky_mask    bool, network resolution
$OUT/stage_b/tri_veto4/<stem>.json detector scalars (s_hat, tau, eps, ...), focal, geometry, provenance
$OUT/stage_b/tri_veto4/manifest.parquet (+ manifest.csv)   one row per image, with QA flags
$OUT/stage_b/tri_veto4/viz/<stem>.png   2x4 diagnostic sheet (docs/viz_guide.md)
```

### 6.4 Using the weights

```python
import numpy as np
from illum_baseline.outputs import pair_weights

z = np.load("$OUT/stage_b/tri_veto4/my_image.npz")
w = z["w"].astype(np.float32)     # (H, W) like the TIFF, stored frame; 1 = full smoothness, 0.05 = relaxed
w_x, w_y = pair_weights(w)        # (H, W-1), (H-1, W): min of horizontal / vertical neighbours
# e.g. smoothness loss on an illumination map I (H, W, C):
#   loss = (w_x[..., None] * (I[:, 1:] - I[:, :-1])**2).mean() + (w_y[..., None] * (I[1:] - I[:-1])**2).mean()
```

- **Metric depth, if needed:** $Z\,[\mathrm{m}] = f_n\,D/300$, where $f_n$ is `focal.f_n` in the Stage A JSON and $D$ is the network-resolution depth. The weights do not depend on $f$.
- **Do not differentiate the upsampled `w`.** Use `w_net` / `q` at network resolution for any further derivative computations.

## 7. Where the changes live

| area | files |
|---|---|
| Detector (all options; pure numpy/scipy) | `illum_baseline/detector.py`: `trilateral`, `gaussian_gradient`, scale-band / veto inside `compute_weights`; also the T4 $\tau$ floor (`threshold_floor`) and $\varepsilon$ floor |
| Config and CLI | `illum_baseline/config.py` (new keys and validation; `load_config` resolves relative paths against the YAML; `apply_sets`, `apply_paths`, `detector_params`); `illum_baseline/cli.py` (`infer`, shared `--config/--set/--input/--output`, `--stems`, automatic detector/sky flags, `diag`, `diag-compare`, `diag-generalize`); `illum_baseline/__main__.py` (`python -m illum_baseline`) |
| Stage A variants | `illum_baseline/preprocess.py` (exposure modes, dither); `illum_baseline/depth.py` (TTA); `illum_baseline/stage_a.py` (`paths.stage_a_variant` → `stage_a/<variant>/`); `illum_baseline/calib.py` (GeoCalib reseeding) |
| Stage B | `illum_baseline/stage_b.py`: inherits a variant's Stage A settings; one parameter set per run directory; records new parameters only when they are on |
| Diagnostics | `illum_baseline/diag.py`: T1 stripe diagnostics, variant metrics, generalization reports |
| Configs | `configs/default.yaml` (= `candidate_tri_veto4.yaml`), `configs/baseline_v0.yaml` |
| Tests | `tests/test_detector.py` (planes exact, scale invariance, ripple attenuation, curvature veto, floors); `tests/test_baseline_repro.py` (baseline_v0 bit-for-bit); `tests/test_diag.py`; `tests/test_cli.py` (path precedence and resolution, `infer` errors) |
