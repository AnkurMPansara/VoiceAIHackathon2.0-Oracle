# Best-Time-to-Call: Data Dictionary (v4)

**Window:** 1 Apr to 30 Sep 2026 · 530,748 dialed attempts · 149,363 sellers
**Sources (IndiaMART warehouse, via Redash):** `data_hotlead_disposition_dtl` (call attempts), `dim_glusr_usr` + `dim_city_master` (address), `fact_glusr_gst_details` (business details), `fact_eto_trd_alert_v2` + `dim_glcat_mcat` (categories).

The dataset is two files linked by the seller GLID (`fk_glusr_usr_id`):

1. **`best_time_to_call_attempts_apr_sep_2026.csv`**: one row per call attempt
2. **`best_time_to_call_sellers.csv`**: one row per seller (address, business type, turnover, categories)

Join them on `fk_glusr_usr_id` to get every attempt with its seller's details.

---

## File 1: Attempts (one row per call attempt)

| Column | Meaning | Fill |
|---|---|---|
| `data_hotlead_disposition_dtlid` | Unique id of this attempt | 100% |
| `fk_glusr_usr_id` | Seller GLID (join key) | 100% |
| `redis_bucket` | Hot-lead type (trigger category): PIM, PUA, UA, PAM, OLP, SCHD, PUT (Power User Top-3), NUR, PANF, ENQR, PNSM, PNSR, PNCHF, OLPR, NVGT. Dashboards group NUR / UA / PUA / PIM as "Rest", others as "Top 3" | 100% |
| `lead_bot_version` | Bot / vendor: `main_vani` (SquadStack), `arrowhead`, plus campaign variants (truncated at 10 characters in the source) | 98.7% |
| `call_attempt_count` | Attempt number (1, 2, 3…). Goes back to 1 when a new hot lead starts for the seller | 99.2% |
| `lead_sent_time` | When the attempt was sent to the dialer | 100% |
| `call_start_time` | **Actual dial time of the attempt** | 98.7% |
| `vendor_response_time` | When the vendor reported back | 99.5% |
| `lead_call_status` | Answered / NotAnswered | 100% |
| `lead_call_duration` | Call duration | 98.7% |
| `disposition_label` | Meeting Fixed, Not Interested, General (talked), Call Later / Busy, Not Answered | 100% |
| `meeting_fixed` | 1 if this attempt fixed a meeting | 100% |
| `lead_tbro_time` | "To Be Reached Out" time: the meeting slot (if a meeting was fixed), the callback time (Call Later / Busy), or the scheduled next retry (not answered) | 70% |

Outcome mix: Not Answered 261,293 · Not Interested 116,364 · General 78,365 · Call Later / Busy 43,672 · Meeting Fixed 31,054

## File 2: Sellers (one row per seller)

| Column | Meaning | Fill |
|---|---|---|
| `fk_glusr_usr_id` | Seller GLID (join key) | 100% |
| `seller_city`, `seller_district`, `seller_state`, `seller_pincode` | Registered address | ~100% |
| `seller_locality` | Locality | 39% |
| `seller_latitude`, `seller_longitude` | Coordinates | 48% |
| `business_type` | Legal status: Proprietorship (121k), Partnership (16k), Limited Company (11k), Others | 98% |
| `annual_turnover` | GST turnover slab: 0–40 L, 40 L–1.5 Cr, 1.5–5 Cr, 5–25 Cr, 25–100 Cr, 100–500 Cr, NA | 98% |
| `nature_of_business` | Manufacturer, Trader – Retailer, Trader – Wholesaler/Distributor, Retailer, Service Provider, etc. | 96% |
| `nature_of_business_secondary` | Secondary business activities from GST (comma-separated) | 98% |
| `gst_registration_year` | Year of GST registration (rough indicator of business age) | 98% |
| `top_category_1` / `_2` / `_3` | Seller's top 3 live product categories (MCAT), by the seller's own preference rank | 97% / 94% / 88% |
| `top_parent_category` | Parent category (PMCAT) of the top category | 79% |
| `top_category_group` | Broad industry group of the top category (e.g. Apparel, Building Material, FMCG) | 97% |
| `num_categories` | Number of live categories the seller deals in | 97% |

---

## Useful derivations

- Current trigger delay: `call_start_time − lead_sent_time` for `call_attempt_count = 1`.
- Retry gap: next attempt's `call_start_time` minus this one's, for the same seller, while `call_attempt_count` keeps increasing.
- Planned vs actual retry: `lead_tbro_time` vs the next attempt's `call_start_time`.
- Patterns by segment: answer rate and meeting-fix rate by hour of `call_start_time` × state, city, business type, turnover, nature of business or category group.

## Notes

- No failure-reason field exists in the source, so retry-by-reason uses `disposition_label`.
- Address is the seller's registered address on IndiaMART.
- **PII:** phone numbers, recordings and call summaries are excluded. Pincode, locality and coordinates are included; drop or coarsen them before sharing outside IndiaMART.
