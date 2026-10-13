# Component 4 – PP1 Work Breakdown

## Project
**Smart Product Comparison Assistant**

## Individual Component
**Component 4 – Personalized and Explainable Product Ranking**

## Target
Complete the core individual Component 4 implementation and prepare a working, testable, evidence-supported demo before **21 October 2026**.

---

# 1. Component 4 Scope

Component 4 combines user preferences with structured outputs from Components 1–3 to generate a personalized product ranking.

The component must:

- generate a personalized product ranking;
- model **User-Preference Uncertainty (UP)**;
- model **Review-Evidence Uncertainty (UE)**;
- analyse ranking stability under uncertainty;
- identify whether the ranking is **Preference-sensitive, Evidence-sensitive, Mixed, or Stable**;
- select the appropriate response: **Clarify, Warn, Combine, or Recommend**;
- explain why a product is ranked highly;
- provide counterfactual support showing how preference changes may alter the recommendation.

### Main C4 Flow

`User Preferences + C1–C3 Outputs → Feature Integration → Personalized Ranking → UP + UE Modelling → Monte Carlo Simulation → Ranking Stability Analysis → Source Identification → Clarify / Warn / Combine / Recommend → Explanation + Counterfactual Support`

---

