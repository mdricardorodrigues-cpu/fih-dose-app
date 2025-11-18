
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="FIH Dose Explorer – PK, AUC, Cmax, NOAEL, MABEL & Cases", layout="wide")

st.title("First-in-Human Dose Explorer")

st.markdown(
    """
    Explore how **dose**, **PK parameters**, **species NOAELs**, **MABEL** and **mechanism-specific risks**
    interact when planning a first-in-human starting dose and escalation strategy.

    Use the tabs to explore:
    - PK / AUC / Cmax
    - Species & NOAEL
    - MABEL vs NOAEL
    - Immune activation
    - Exposure margins
    - RPL case examples + FIH protocol generator
    """
)

# ----------------- GLOBAL SIDEBAR CONTROLS -----------------
st.sidebar.header("Global PK parameters")

dose = st.sidebar.slider("Dose (mg)", 1.0, 1000.0, 100.0, 10.0)
tau = st.sidebar.slider("Dosing interval (hours)", 4.0, 48.0, 24.0, 4.0)
t_half = st.sidebar.slider("Half-life (hours)", 1.0, 72.0, 12.0, 1.0)
Vd = st.sidebar.slider("Volume of distribution (L)", 5.0, 200.0, 50.0, 5.0)
F = st.sidebar.slider("Bioavailability F", 0.1, 1.0, 1.0, 0.05)

ke = np.log(2) / t_half  # elimination rate constant

# Time grid
t_end = 5 * tau
t = np.linspace(0, t_end, 1000)


def pk_conc(t, dose, ke, Vd, tau, F, n_doses=10):
    conc = np.zeros_like(t)
    for n in range(n_doses):
        t_dose = n * tau
        contrib = np.where(
            t >= t_dose,
            (F * dose / Vd) * np.exp(-ke * (t - t_dose)),
            0.0,
        )
        conc += contrib
    return conc


C = pk_conc(t, dose, ke, Vd, tau, F)
Cmax = C.max()
t_Cmax = t[np.argmax(C)]
AUC = np.trapz(C, t)

# ----------------- SPECIES DATA (shared across tabs) -----------------
species_data = {
    "Rat": {"NOAEL_mgkg": 50, "AUC": 250},
    "Dog": {"NOAEL_mgkg": 10, "AUC": 120},
    "Monkey": {"NOAEL_mgkg": 5, "AUC": 80},
}
km_values = {"Rat": 6, "Dog": 20, "Monkey": 12}  # body surface area conversion factors

species_names = list(species_data.keys())
NOAELs = [species_data[s]["NOAEL_mgkg"] for s in species_names]
AUCs_species = [species_data[s]["AUC"] for s in species_names]

# ----------------- TABS -----------------
tab_pk, tab_species, tab_mabel, tab_immune, tab_margin, tab_cases = st.tabs(
    [
        "PK / AUC / Cmax",
        "Species & NOAEL",
        "MABEL vs NOAEL",
        "Immune Activation",
        "Exposure Margins",
        "RPL Cases + FIH Generator",
    ]
)

# ----------------- TAB 1: PK / AUC / Cmax -----------------
with tab_pk:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Concentration–time profile")

        fig, ax = plt.subplots()
        ax.plot(t, C)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title("Multiple-dose PK – one-compartment")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("AUC and Cmax vs dose")

        # Explore a set of doses to show trend
        dose_grid = np.linspace(max(1.0, dose / 10.0), dose * 2.0, 6)
        AUCs = []
        Cmaxs = []
        for d in dose_grid:
            Csim = pk_conc(t, d, ke, Vd, tau, F)
            AUCs.append(np.trapz(Csim, t))
            Cmaxs.append(Csim.max())

        fig2, ax2 = plt.subplots()
        ax2.plot(dose_grid, AUCs, marker="o", label="AUC")
        ax2.plot(dose_grid, Cmaxs, marker="s", label="Cmax")
        ax2.set_xlabel("Dose (mg)")
        ax2.set_ylabel("Exposure")
        ax2.set_title("AUC and Cmax vs dose (current PK assumptions)")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    st.markdown(
        f"""
        **Current settings**

        - Cmax: {Cmax:.2f} mg/L at {t_Cmax:.1f} h  
        - AUC (0–{t_end:.0f} h): {AUC:.2f} mg·h/L  
        - Elimination rate constant ke: {ke:.3f} h⁻¹  
        """
    )

# ----------------- TAB 2: Species & NOAEL -----------------
with tab_species:
    st.subheader("NOAEL and exposure in different species")

    body_weight_human = st.slider("Assumed human body weight (kg)", 40.0, 120.0, 70.0, 5.0)

    HEDs = []
    for s in species_names:
        animal_noael = species_data[s]["NOAEL_mgkg"]
        km_animal = km_values[s]
        km_human = 37
        hed_mgkg = animal_noael * (km_animal / km_human)
        HEDs.append(hed_mgkg)

    hed_dose_mg = [hed * body_weight_human for hed in HEDs]

    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots()
        ax3.bar(species_names, NOAELs)
        ax3.set_ylabel("NOAEL (mg/kg)")
        ax3.set_title("NOAEL by species")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots()
        ax4.bar(species_names, AUCs_species)
        ax4.set_ylabel("AUC at NOAEL (arbitrary units)")
        ax4.set_title("Exposure at NOAEL by species")
        st.pyplot(fig4)

    with col2:
        fig5, ax5 = plt.subplots()
        ax5.bar(species_names, HEDs)
        ax5.set_ylabel("HED (mg/kg)")
        ax5.set_title("Human equivalent dose (HED) from each species")
        st.pyplot(fig5)

        fig6, ax6 = plt.subplots()
        ax6.bar(species_names, hed_dose_mg)
        ax6.set_ylabel("HED (mg) for selected human weight")
        ax6.set_title(f"HED in mg for {body_weight_human:.0f} kg human")
        st.pyplot(fig6)

# ----------------- TAB 3: MABEL vs NOAEL -----------------
with tab_mabel:
    st.subheader("Compare MABEL-based and NOAEL-based starting doses")

    # NOAEL-based MRSD from most sensitive species (by lowest AUC at NOAEL)
    most_sensitive_species = species_names[np.argmin(AUCs_species)]
    most_sensitive_index = species_names.index(most_sensitive_species)

    # HEDs for each species (mg/kg) for a 70 kg human
    body_weight_default = 70.0
    HEDs_local = []
    for s in species_names:
        animal_noael = species_data[s]["NOAEL_mgkg"]
        km_animal = km_values[s]
        km_human = 37
        hed_mgkg = animal_noael * (km_animal / km_human)
        HEDs_local.append(hed_mgkg)
    most_sensitive_hed_mgkg = HEDs_local[most_sensitive_index]

    sf_noael = st.slider("Safety factor for NOAEL-based MRSD", 1.0, 50.0, 10.0, 1.0)
    mrsd_noael_mgkg = most_sensitive_hed_mgkg / sf_noael
    mrsd_noael_mg = mrsd_noael_mgkg * body_weight_default

    st.write(f"Most sensitive species (by lowest AUC at NOAEL): {most_sensitive_species}")
    st.write(f"HED from {most_sensitive_species}: {most_sensitive_hed_mgkg:.2f} mg/kg")
    st.write(f"NOAEL-based MRSD (mg/kg): {mrsd_noael_mgkg:.3f}")
    st.write(f"NOAEL-based MRSD total dose (mg) for {body_weight_default:.0f} kg human: {mrsd_noael_mg:.1f}")

    st.markdown("---")

    st.subheader("MABEL-based dose (target Cmax)")

    target_cmax_mabel = st.slider("Target Cmax for MABEL (mg/L)", 0.01, 5.0, 0.25, 0.01)

    # Approximate dose needed to reach target Cmax in single dose:
    # For a one-compartment model: C0 = F*dose / Vd -> dose ≈ target_cmax * Vd / F
    mabel_dose_single = target_cmax_mabel * Vd / F

    st.write(f"Approximate MABEL-based single dose: {mabel_dose_single:.1f} mg (for chosen Vd and F)")

    # Compare candidate human dose grid against both thresholds
    max_ref = max(mabel_dose_single, mrsd_noael_mg, 1.0)
    dose_grid2 = np.linspace(0, max_ref * 1.5, 50)
    mabel_line = np.full_like(dose_grid2, mabel_dose_single)
    noael_line = np.full_like(dose_grid2, mrsd_noael_mg)

    fig7, ax7 = plt.subplots()
    ax7.plot(dose_grid2, mabel_line, label="MABEL-based", linestyle="--")
    ax7.plot(dose_grid2, noael_line, label="NOAEL-based MRSD", linestyle="-.")
    ax7.set_xlabel("Candidate human dose (mg)")
    ax7.set_ylabel("Threshold dose (mg)")
    ax7.set_title("Comparison of MABEL vs NOAEL-based starting dose")
    ax7.legend()
    ax7.grid(True)
    st.pyplot(fig7)

# ----------------- TAB 4: IMMUNE ACTIVATION -----------------
with tab_immune:
    st.subheader("Simple immune activation pattern")

    k_cyt = st.slider("Immune activation sensitivity", 0.1, 1.5, 0.4, 0.05)
    baseline_marker = 1.0
    max_C = max(Cmax, 1e-6)
    immune_marker = baseline_marker * np.exp(k_cyt * (C / max_C))

    fig8, ax8 = plt.subplots()
    ax8.plot(t, immune_marker)
    ax8.set_xlabel("Time (hours)")
    ax8.set_ylabel("Relative immune activation")
    ax8.set_title("Immune activation vs time (exposure-driven)")
    ax8.grid(True)
    st.pyplot(fig8)

# ----------------- TAB 5: EXPOSURE MARGINS -----------------
with tab_margin:
    st.subheader("Exposure margins: human AUC vs animal NOAEL AUC")

    # Human AUC at current dose vs AUC at NOAEL for each species
    ratios = [AUC / auc_noael for auc_noael in AUCs_species]

    fig9, ax9 = plt.subplots()
    ax9.bar(species_names, ratios)
    ax9.axhline(1.0, color="red", linestyle="--", label="Ratio = 1")
    ax9.set_ylabel("Human AUC / Animal NOAEL AUC")
    ax9.set_title("Exposure margin by species")
    ax9.legend()
    ax9.grid(True, axis="y")
    st.pyplot(fig9)

# ----------------- TAB 6: RPL CASES + FIH GENERATOR -----------------
with tab_cases:
    st.subheader("RPL Case Examples and FIH Protocol Generator")

    st.markdown(
        """
        Each case below represents a different type of first-in-human challenge seen at RPL:
        - A small molecule with QT risk
        - A long-acting siRNA for MASH
        - A one-shot gene-editing therapy for LDL lowering
        - A potent immune modulator where MABEL must dominate

        Use the controls and graphs in each case to choose a **starting dose** and
        think through **escalation strategy**, then use the generator at the bottom to
        build a simple cohort table.
        """
    )

    case_choice = st.selectbox(
        "Choose a case",
        [
            "RPL-QT01 – Small molecule with QT prolongation risk",
            "RPL-SI01 – siRNA for liver target in MASH",
            "RPL-GE01 – Gene editing targeting LDL",
            "RPL-IM01 – Immune modulator (MABEL-driven)",
        ],
    )

    # Initialise variables used later in generator (to avoid reference before assignment)
    start_dose_qt = None
    mabel_dose_si = None
    start_dose_ge = None
    mabel_dose_im = None

    # ----------------- CASE 1: RPL-QT01 -----------------
    if case_choice == "RPL-QT01 – Small molecule with QT prolongation risk":
        st.markdown(
            """
            ### RPL-QT01 – Oral KV-channel blocker with QT risk

            **Scenario summary**

            - Oral once-daily small molecule targeting a neuronal KV channel for neuropathic pain.
            - Preclinical safety shows **dose-dependent QT prolongation**, particularly in dogs.
            - hERG and telemetry suggest risk increases as Cmax approaches a critical threshold.

            In this exercise you are trying to:
            - Pick a starting dose that is clearly below the QT concern threshold.
            - Define a reasonable set of escalation steps that probe exposure without crossing that threshold.
            """
        )

        col_qt1, col_qt2 = st.columns(2)

        with col_qt1:
            qt_threshold = st.number_input(
                "QT concern threshold Cmax (mg/L, free or total equivalent)",
                min_value=0.01,
                max_value=5.0,
                value=0.8,
                step=0.05,
            )
            start_dose_qt = st.slider("Candidate starting dose (mg)", 1.0, 80.0, 10.0, 1.0)
            n_cohorts_qt = st.slider("Number of escalation cohorts", 2, 6, 5, 1)
            escalation_factor_qt = st.slider("Escalation factor per cohort", 1.1, 3.0, 2.0, 0.1)

            doses_qt = [start_dose_qt * (escalation_factor_qt ** i) for i in range(n_cohorts_qt)]

            st.markdown(
                """
                **How to interpret**

                - Cohorts on the left of the graph are early SAD doses.
                - As you increase the escalation factor, later cohorts quickly approach or exceed
                  the QT concern threshold line.
                - Try to find a combination of starting dose and escalation factor that:
                  - Gives you informative exposure increases.
                  - Keeps all Cmax values below or only just touching the QT concern threshold.
                """
            )

        with col_qt2:
            cmax_list = []
            for d in doses_qt:
                Csim = pk_conc(t, d, ke, Vd, tau, F)
                cmax_list.append(Csim.max())

            fig_qt, ax_qt = plt.subplots()
            ax_qt.bar(range(1, n_cohorts_qt + 1), cmax_list)
            ax_qt.axhline(qt_threshold, color="red", linestyle="--", label="QT concern threshold")
            ax_qt.set_xlabel("Cohort")
            ax_qt.set_ylabel("Cmax (mg/L)")
            ax_qt.set_title("Cmax per cohort vs QT concern threshold")
            ax_qt.legend()
            st.pyplot(fig_qt)

    # ----------------- CASE 2: RPL-SI01 -----------------
    elif case_choice == "RPL-SI01 – siRNA for liver target in MASH":
        st.markdown(
            """
            ### RPL-SI01 – GalNAc-siRNA for a liver target in MASH

            **Scenario summary**

            - GalNAc-siRNA delivered to hepatocytes, with target mRNA knockdown lasting weeks to months.
            - NHP studies show a steep dose–knockdown relationship with a long duration of effect.
            - FIH is planned directly in patients with MASH, not healthy volunteers.

            In this exercise you are trying to:
            - Link a desired percentage of liver knockdown to a suitable **MABEL-based starting dose**.
            - Visualise how small changes in dose can move you from minimal to near-maximal knockdown.
            """
        )

        col_si1, col_si2 = st.columns(2)

        with col_si1:
            target_knockdown = st.slider("Target knockdown for first cohort (%)", 10, 60, 20, 5)
            ec50_si = st.number_input(
                "Approximate EC50 for knockdown (mg)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.5,
            )
            hill_si = st.slider("Hill coefficient for knockdown curve", 0.5, 3.0, 1.0, 0.1)

            dose_grid_si = np.linspace(0.1, 200.0, 1000)
            effect_grid_si = 100.0 * (dose_grid_si ** hill_si) / (
                (ec50_si ** hill_si) + (dose_grid_si ** hill_si)
            )
            idx = np.where(effect_grid_si >= target_knockdown)[0]
            if len(idx) > 0:
                mabel_dose_si = dose_grid_si[idx[0]]
            else:
                mabel_dose_si = dose_grid_si[-1]

            st.write(f"Approximate dose for {target_knockdown}% knockdown: {mabel_dose_si:.1f} mg")

            # Design cohort doses around this MABEL estimate
            start_dose_si = st.slider(
                "Starting dose for siRNA cohorts (mg)",
                0.1,
                float(max(1.0, mabel_dose_si * 2.0)),
                float(max(0.1, mabel_dose_si / 2.0)),
                0.1,
            )
            n_cohorts_si = st.slider("Number of siRNA cohorts", 2, 6, 4, 1)
            esc_factor_si = st.slider("Escalation factor per siRNA cohort", 1.1, 3.0, 1.7, 0.1)

            doses_si = [start_dose_si * (esc_factor_si ** i) for i in range(n_cohorts_si)]

            st.markdown(
                """
                **Teaching points**

                - A very high knockdown (e.g. >70–80%) might not be needed in the first cohort.
                - A modest knockdown target (e.g. 20–30%) usually gives a safer FIH starting dose.
                - You can now see knockdown per cohort directly, not just at a single MABEL dose.
                """
            )

        with col_si2:
            fig_si, ax_si = plt.subplots()
            ax_si.plot(dose_grid_si, effect_grid_si)
            ax_si.axhline(target_knockdown, color="grey", linestyle="--", label="Target knockdown")
            ax_si.set_xlabel("Single dose (mg)")
            ax_si.set_ylabel("Liver target knockdown (%)")
            ax_si.set_title("Dose–knockdown curve for RPL-SI01")
            ax_si.legend()
            ax_si.grid(True)
            st.pyplot(fig_si)

        # New: graph cohort doses vs corresponding knockdown
        knockdown_per_cohort = []
        for d in doses_si:
            eff = 100.0 * (d ** hill_si) / ((ec50_si ** hill_si) + (d ** hill_si))
            knockdown_per_cohort.append(eff)

        st.markdown("#### Cohort doses vs expected liver knockdown")

        fig_si_coh, ax_si_coh = plt.subplots()
        ax_si_coh.plot(range(1, n_cohorts_si + 1), doses_si, marker="o", label="Dose (mg)")
        ax_si_coh_2 = ax_si_coh.twinx()
        ax_si_coh_2.plot(
            range(1, n_cohorts_si + 1),
            knockdown_per_cohort,
            marker="s",
            linestyle="--",
            label="Knockdown (%)",
        )
        ax_si_coh.set_xlabel("Cohort")
        ax_si_coh.set_ylabel("Dose (mg)")
        ax_si_coh_2.set_ylabel("Liver knockdown (%)")
        ax_si_coh.set_title("RPL-SI01 – Dose and expected liver knockdown per cohort")
        ax_si_coh.grid(True)

        # Add a combined legend
        lines_1, labels_1 = ax_si_coh.get_legend_handles_labels()
        lines_2, labels_2 = ax_si_coh_2.get_legend_handles_labels()
        ax_si_coh.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        st.pyplot(fig_si_coh)

    # ----------------- CASE 3: RPL-GE01 -----------------
    elif case_choice == "RPL-GE01 – Gene editing targeting LDL":
        st.markdown(
            """
            ### RPL-GE01 – In vivo gene editing to lower LDL

            **Scenario summary**

            - LNP-formulated CRISPR-based therapy targeting a liver gene involved in LDL regulation.
            - Single IV dose aimed at **permanent** LDL-C reduction via hepatocyte editing.
            - NHP data show dose-dependent editing and LDL lowering, with off-target events at higher doses.

            In this exercise you are trying to:
            - Identify a minimal effective dose that gives clinically meaningful LDL-C reduction.
            - Choose a FIH starting dose **below** this minimal effective dose.
            - Think about when to stop escalating once an adequate LDL reduction is observed.
            """
        )

        col_ge1, col_ge2 = st.columns(2)

        with col_ge1:
            target_ldl_reduction = st.slider(
                "Target LDL-C reduction for first effective cohort (%)", 20, 80, 30, 5
            )
            ed50_ge = st.number_input(
                "Approximate ED50 for LDL-C reduction (mg)",
                min_value=0.1,
                max_value=200.0,
                value=20.0,
                step=1.0,
            )
            hill_ge = st.slider("Hill coefficient for LDL-C reduction curve", 0.5, 3.0, 1.2, 0.1)

            dose_grid_ge = np.linspace(0.5, 300.0, 1000)
            effect_grid_ge = 100.0 * (dose_grid_ge ** hill_ge) / (
                (ed50_ge ** hill_ge) + (dose_grid_ge ** hill_ge)
            )
            idx_ge = np.where(effect_grid_ge >= target_ldl_reduction)[0]
            if len(idx_ge) > 0:
                min_effective_dose_ge = dose_grid_ge[idx_ge[0]]
            else:
                min_effective_dose_ge = dose_grid_ge[-1]

            st.write(
                f"Approximate minimal dose for {target_ldl_reduction}% LDL-C reduction: "
                f"{min_effective_dose_ge:.1f} mg"
            )

            start_dose_ge = st.slider(
                "Choose starting dose as a fraction of this (mg)",
                0.1,
                float(min_effective_dose_ge),
                float(min_effective_dose_ge / 3.0),
                0.1,
            )

            n_cohorts_ge = st.slider("Number of gene-editing cohorts", 2, 6, 4, 1)
            esc_factor_ge = st.slider("Escalation factor per gene-editing cohort", 1.1, 3.0, 1.7, 0.1)

            doses_ge = [start_dose_ge * (esc_factor_ge ** i) for i in range(n_cohorts_ge)]

            st.markdown(
                """
                **Teaching points**

                - For irreversible therapies, a small but meaningful LDL-C reduction can be enough in FIH.
                - You rarely need to chase the plateau of the curve in early cohorts.
                - Stopping escalation once robust LDL lowering is shown can be safer than pushing to the top.
                """
            )

        with col_ge2:
            fig_ge, ax_ge = plt.subplots()
            ax_ge.plot(dose_grid_ge, effect_grid_ge, label="LDL-C reduction curve")
            ax_ge.axvline(min_effective_dose_ge, color="green", linestyle="--", label="Minimal effective dose")
            ax_ge.axvline(start_dose_ge, color="blue", linestyle=":", label="Chosen starting dose")
            ax_ge.set_xlabel("Single dose (mg)")
            ax_ge.set_ylabel("LDL-C reduction (%)")
            ax_ge.set_title("Dose–response for RPL-GE01")
            ax_ge.legend()
            ax_ge.grid(True)
            st.pyplot(fig_ge)

        # New: cohort doses vs expected LDL reduction
        ldl_reduction_per_cohort = []
        for d in doses_ge:
            eff_ge = 100.0 * (d ** hill_ge) / ((ed50_ge ** hill_ge) + (d ** hill_ge))
            ldl_reduction_per_cohort.append(eff_ge)

        st.markdown("#### Cohort doses vs expected LDL-C reduction")

        fig_ge_coh, ax_ge_coh = plt.subplots()
        ax_ge_coh.plot(range(1, n_cohorts_ge + 1), doses_ge, marker="o", label="Dose (mg)")
        ax_ge_coh_2 = ax_ge_coh.twinx()
        ax_ge_coh_2.plot(
            range(1, n_cohorts_ge + 1),
            ldl_reduction_per_cohort,
            marker="s",
            linestyle="--",
            label="LDL-C reduction (%)",
        )
        ax_ge_coh.set_xlabel("Cohort")
        ax_ge_coh.set_ylabel("Dose (mg)")
        ax_ge_coh_2.set_ylabel("LDL-C reduction (%)")
        ax_ge_coh.set_title("RPL-GE01 – Dose and expected LDL-C reduction per cohort")
        ax_ge_coh.grid(True)

        lines_1_ge, labels_1_ge = ax_ge_coh.get_legend_handles_labels()
        lines_2_ge, labels_2_ge = ax_ge_coh_2.get_legend_handles_labels()
        ax_ge_coh.legend(lines_1_ge + lines_2_ge, labels_1_ge + labels_2_ge, loc="upper left")

        st.pyplot(fig_ge_coh)

    # ----------------- CASE 4: RPL-IM01 -----------------
    elif case_choice == "RPL-IM01 – Immune modulator (MABEL-driven)":
        st.markdown(
            """
            ### RPL-IM01 – CD3-engaging immune modulator

            **Scenario summary**

            - Bispecific or agonist engaging CD3 on T cells and a tumour antigen.
            - Very steep human PD: small exposure increases can trigger large cytokine surges.
            - NHP NOAEL is relatively high, but human PBMC assays show cytokine EC10 at low ng/mL.

            In this exercise you are trying to:
            - Use a PD-based, MABEL-like approach to pick a starting dose far below EC10.
            - See how quickly cytokine activation rises as dose increases.
            - Compare a MABEL-based starting dose with a hypothetical NOAEL-based MRSD.
            """
        )

        col_im1, col_im2 = st.columns(2)

        with col_im1:
            ec10_cyt = st.number_input(
                "EC10 for cytokine release (ng/mL, conceptual)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.05,
            )
            max_cyt_allowed = st.slider(
                "Acceptable relative cytokine level at FIH dose (fraction of EC10)",
                0.05,
                1.0,
                0.2,
                0.05,
            )
            dose_grid_im = np.linspace(0.001, 1.0, 300)  # mg range as a proxy

            ec50_im = ec10_cyt * 5.0
            hill_im = 3.0
            conc_grid_im = dose_grid_im  # treat dose as proportional to concentration
            pd_cyt = (conc_grid_im ** hill_im) / ((ec50_im ** hill_im) + (conc_grid_im ** hill_im))

            pd_at_ec10 = (ec10_cyt ** hill_im) / ((ec50_im ** hill_im) + (ec10_cyt ** hill_im))
            target_pd = max_cyt_allowed * pd_at_ec10
            idx_im = np.where(pd_cyt >= target_pd)[0]
            if len(idx_im) > 0:
                mabel_dose_im = dose_grid_im[idx_im[0]]
            else:
                mabel_dose_im = dose_grid_im[-1]

            st.write(f"Approximate MABEL-like starting dose (arbitrary mg units): {mabel_dose_im:.4f} mg")

            # Hypothetical NOAEL-based MRSD for comparison
            noael_based_dose_im = st.number_input(
                "Hypothetical NOAEL-based MRSD (mg, arbitrary scale)",
                min_value=0.001,
                max_value=10.0,
                value=0.5,
                step=0.01,
            )

            # Design cohort doses around MABEL-like starting dose
            start_dose_im = st.slider(
                "Starting dose for immune-modulator cohorts (mg, arbitrary scale)",
                0.0005,
                float(max(0.01, noael_based_dose_im)),
                float(max(0.0005, mabel_dose_im / 2.0)),
                0.0005,
            )
            n_cohorts_im = st.slider("Number of immune-modulator cohorts", 2, 6, 4, 1)
            esc_factor_im = st.slider("Escalation factor per immune-modulator cohort", 1.1, 3.0, 1.5, 0.1)

            doses_im = [start_dose_im * (esc_factor_im ** i) for i in range(n_cohorts_im)]

        with col_im2:
            fig_im, ax_im = plt.subplots()
            ax_im.plot(dose_grid_im, pd_cyt, label="Cytokine activation curve (arbitrary)")
            ax_im.axvline(mabel_dose_im, color="blue", linestyle="--", label="MABEL-like FIH dose")
            ax_im.axvline(noael_based_dose_im, color="red", linestyle=":", label="NOAEL-based MRSD")
            ax_im.set_xlabel("Dose (mg, arbitrary scale)")
            ax_im.set_ylabel("Relative cytokine activation")
            ax_im.set_title("Dose–cytokine relationship for RPL-IM01")
            ax_im.legend()
            ax_im.grid(True)
            st.pyplot(fig_im)

            st.markdown(
                """
                **MABEL vs NOAEL explanation**

                - The **MABEL-like dose** is chosen to keep cytokine activation well below EC10 based on
                  human PBMC data.
                - A **NOAEL-based MRSD** might appear safe in animals but, if used directly, could place
                  many humans on the steep part of the cytokine curve.
                - The graph shows how the NOAEL-based dose can sit much further to the right on a very steep
                  PD curve, where small errors in prediction may lead to large clinical effects.
                """
            )

        # New: cohort doses vs cytokine activation
        cytokine_per_cohort = []
        for d in doses_im:
            eff_im = (d ** hill_im) / ((ec50_im ** hill_im) + (d ** hill_im))
            cytokine_per_cohort.append(eff_im / pd_at_ec10 * 100.0)  # as % of EC10 PD

        st.markdown("#### Cohort doses vs relative cytokine activation")

        fig_im_coh, ax_im_coh = plt.subplots()
        ax_im_coh.plot(range(1, n_cohorts_im + 1), doses_im, marker="o", label="Dose (mg, arbitrary)")
        ax_im_coh_2 = ax_im_coh.twinx()
        ax_im_coh_2.plot(
            range(1, n_cohorts_im + 1),
            cytokine_per_cohort,
            marker="s",
            linestyle="--",
            label="Cytokine activation (% of EC10 PD)",
        )
        ax_im_coh.set_xlabel("Cohort")
        ax_im_coh.set_ylabel("Dose (mg, arbitrary)")
        ax_im_coh_2.set_ylabel("Cytokine activation (% of EC10 PD)")
        ax_im_coh.set_title("RPL-IM01 – Dose and relative cytokine activation per cohort")
        ax_im_coh.grid(True)

        lines_1_im, labels_1_im = ax_im_coh.get_legend_handles_labels()
        lines_2_im, labels_2_im = ax_im_coh_2.get_legend_handles_labels()
        ax_im_coh.legend(lines_1_im + lines_2_im, labels_1_im + labels_2_im, loc="upper left")

        st.pyplot(fig_im_coh)

    # ----------------- FIH PROTOCOL GENERATOR -----------------
    st.markdown("---")
    st.subheader("FIH Protocol Dose Generator")

    gen_case = st.selectbox(
        "Select which case to base the protocol on",
        [
            "Generic",
            "Use RPL-QT01 current settings",
            "Use RPL-SI01 suggestion",
            "Use RPL-GE01 suggestion",
            "Use RPL-IM01 suggestion",
        ],
    )

    # Defaults for generator
    if gen_case == "Use RPL-QT01 current settings" and start_dose_qt is not None:
        base_start = start_dose_qt
    elif gen_case == "Use RPL-SI01 suggestion" and mabel_dose_si is not None:
        base_start = float(mabel_dose_si / 2.0)
    elif gen_case == "Use RPL-GE01 suggestion" and start_dose_ge is not None:
        base_start = float(start_dose_ge)
    elif gen_case == "Use RPL-IM01 suggestion" and mabel_dose_im is not None:
        base_start = float(mabel_dose_im / 5.0)
    else:
        base_start = 10.0

    start_dose_gen = st.number_input(
        "Starting dose for protocol (mg)",
        min_value=0.0001,
        max_value=1000.0,
        value=base_start,
        format="%.4f",
    )
    n_cohorts_gen = st.slider("Number of cohorts", 2, 10, 5, 1)
    esc_factor_gen = st.slider("Escalation factor", 1.1, 4.0, 2.0, 0.1)
    pop_type = st.selectbox("Population", ["Healthy volunteers", "Patients"])
    sentinel = st.checkbox("Use sentinel dosing in each cohort", value=True)

    doses_protocol = [start_dose_gen * (esc_factor_gen ** i) for i in range(n_cohorts_gen)]
    rows = []
    for i, dprot in enumerate(doses_protocol, start=1):
        row = {
            "Cohort": i,
            "Dose (mg)": round(dprot, 4),
            "Population": pop_type,
            "Sentinel dosing": "Yes" if sentinel else "No",
        }
        rows.append(row)

    st.write("Proposed FIH dose-escalation table:")
    st.table(rows)

    # Plot the protocol doses
    fig_gen, ax_gen = plt.subplots()
    ax_gen.plot(range(1, n_cohorts_gen + 1), doses_protocol, marker="o")
    ax_gen.set_xlabel("Cohort")
    ax_gen.set_ylabel("Dose (mg)")
    ax_gen.set_title("FIH protocol dose escalation by cohort")
    ax_gen.grid(True)
    st.pyplot(fig_gen)
