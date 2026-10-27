# SORA LaTeX Report Generator (`sora_latex_report`)

Gerador automático de relatórios científicos em formato **LaTeX (`.tex`)** para análises de ocultações estelares realizadas com a biblioteca **[SORA (Stellar Occultation Reduction and Analysis)](https://sora.readthedocs.io/)**.

---

## Funcionalidades

A função `generate_sora_latex_report` extrai diretamente dos objetos do SORA (`Occultation`, `Star`, `Body`, `LightCurve`, `Observer` e `ChiSquare`) todos os parâmetros necessários e gera:

1. **Tabelas em LaTeX prontas para compilação (Overleaf / `pdflatex`)**:
   - **Tabela 1**: Dados astrométricos, fotométricos e diâmetro da estrela ocultada tanto em **miliarcosegundos (`mas`)** quanto em **quilômetros projetados (`km`)** (Kervella et al. 2004; Gaia DR3) + parâmetros físicos e efeméride do corpo ocultante.
   - **Tabela 2**: Circunstâncias geocêntricas da ocultação (instante de máxima aproximação, velocidade da sombra, distância, *closest approach*, ângulo de posição, número de cordas positivas/negativas e links clicáveis para a predição no **Lucky Star** e para o evento no **Occultation Portal**).
   - **Tabela 3**: Circunstâncias observacionais detalhadas de todas as estações participantes (coordenadas geodésicas convertidas para Graus/Minutos/Segundos, altitude, observadores, abertura, detector, **tempo de exposição (`Exp. Time`)**, **tempo de ciclo (`Cycle Time`)**, filtro e status).
   - **Tabelas da Seção 2**: Resultados do ajuste de imersão e emersão para cada curva de luz positiva ($\chi^2_{\min}$, $N_{\text{pts}}$, $\chi^2_{\text{pdf}}$, dispersão, banda espectral, diâmetro estelar em `km` e `mas`, e velocidade radial projetada $v_{\text{rad}}$), além de tabela dedicada para as **cordas negativas** com seu passo temporal equivalente em `km`.
   - **Tabelas da Seção 3**: Resumo dos instantes ($1\sigma$), melhores parâmetros da elipse ajustada (`Center_f`, `Center_g`, `R_equat`, `Oblateness` e `Position Angle`) e **Posição Astrométrica resultante (`new_astrometric_position`)** com offsets geocêntricos ($\Delta\alpha\cos\delta$, $\Delta\delta$) e incertezas em `mas` e `km`.
   - **Listing 1**: Saída completa do log do SORA (`print(occ)`).

2. **Figuras compostas salvas automaticamente em alta resolução (`300 dpi`)**:
   - Mapa da predição da ocultação (`pred_map.png`).
   - Painéis $2\times 2$ para cada observação positiva contendo as curvas de luz normalizadas + modelo geométrico/Fresnel/diâmetro estelar (topo) e curvas de $\chi^2$ de imersão e emersão (base).
   - Gráficos das curvas de luz para as **cordas negativas**.
   - Projeção das cordas positivas (com barras de erro em vermelho), cordas negativas, melhor elipse ajustada (com envelope de incerteza de $1\sigma$ ou $3\sigma$ em cinza) e **ponto central da elipse ajustada `(Center_f, Center_g)`** com barras de erro de $1\sigma$.
   - Painel com os 5 gráficos de $\chi^2$ dos parâmetros da elipse ajustada.

---

## Pré-requisitos

- Python 3.8+
- `sora` (Stellar Occultation Reduction and Analysis)
- `numpy`, `matplotlib`, `astropy`

---

## Como Usar no Jupyter Notebook

Basta colocar o arquivo `sora_latex_report.py` na mesma pasta do seu notebook (ou adicioná-lo ao `sys.path`) e chamar a função na última célula do notebook, logo após ajustar a elipse com `occ.fit_ellipse(...)`:

### 1. Uso rápido (com links do Lucky Star / Occultation Portal)

Se você seguiu a convenção padrão de variáveis nos seus notebooks (`lc1`, `lc2`..., `obs1_chi2` ou `obs1_chi2_i` / `obs1_chi2_e`, e `ellipse_chi2`), a função detecta todas as variáveis automaticamente da memória do Jupyter:

```python
from sora_latex_report import generate_sora_latex_report

generate_sora_latex_report(
    occ,
    lucky_star_url="https://lesia.obspm.fr/lucky-star/occ.php?p=...",
    occultation_portal_url="https://occultation.tug.tubitak.gov.tr/...",
)
```

### 2. Uso completo (personalizando metadados dos observadores e arquivos de saída)

```python
from sora_latex_report import generate_sora_latex_report

# (Opcional) Informações adicionais de telescópio, câmera e filtro para a Tabela 3
observer_metadata = {
    "Observador1": {
        "site_name": "Nome do Observatório",
        "observers": "Nome Sobrenome",
        "aperture": "300.0",
        "detector": "QHY174M-GPS",
        "filter": "Clear",
        "status": "Positive",
    },
}

generate_sora_latex_report(
    occ=occ,
    ellipse_chi2=ellipse_chi2,
    observer_metadata=observer_metadata,
    lucky_star_url="https://lesia.obspm.fr/lucky-star/occ.php?p=...",
    occultation_portal_url="https://occultation.tug.tubitak.gov.tr/...",
    author="João David Moreira",
    preliminary=True,
    pred_map_file="pred_map.png",
    output_tex="relatorio_ocultacao.tex",
    figures_dir="figures_report",
)
```

---

## Compilando no Overleaf

1. Crie um novo projeto no [Overleaf](https://www.overleaf.com/).
2. Faça upload do arquivo `.tex` gerado (ex.: `relatorio_ocultacao.tex`), da imagem `pred_map.png` e da pasta de figuras gerada (`figures_report/`).
3. Clique em **Recompile**.
