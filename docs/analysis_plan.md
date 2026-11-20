# Analysis Plan (written before looking at results)

**Experiment:** 64,000 customers who bought in the last 12 months were randomly split into three equal groups: Mens E-Mail, Womens E-Mail, or No E-Mail (control). Outcomes were tracked for two weeks.

**Decisions this analysis informs:**
1. Do the emails work, and which one works better?
2. Should the company email everyone, or only the customers most likely to respond?

## Metrics
- **Primary:** website visit within 2 weeks
- **Secondary:** conversion (made a purchase) and spend per customer

## Hypotheses
- **H0 (null):** an email has no effect on the metric compared with No E-Mail.
- **H1 (alternative):** the email changes the metric (two-sided).

## Rules I'm committing to now
- Alpha = 0.05, two-sided.
- 4 tests (2 emails × visit and conversion), so I'll use a Holm correction.
- Spend is skewed (most customers spend $0), so I'll use a bootstrap instead of a z-test.
- Before testing: check the three-way split (SRM, flag if p < 0.01) and check that customer features are balanced across groups.
- **Machine learning uses only features known before the email was sent.** Outcomes (visit, conversion, spend) are never used as features.
- Every ML model is judged on a held-out test set (30% of customers) and compared with a simple baseline.